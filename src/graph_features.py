import os
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from attention import aggregate_attention_layers, load_attns
from visualization import plot_attention_matrix, plot_features
from utils import relative_to_absolute_path
from args import get_args
import math
import seaborn as sns
from graph_metrics import *
from prompts import Prompts

def create_graph_single_attn(attn, **kwargs):
    """
    I am given a single attention matrix (NxN) and I want to create a graph from it.
    Edges are chosen based on the top k values in the attention matrix.

    Selects top k outgoing edges (query -> keys) for each query node.
    """
    mode = kwargs.get("mode", "top_k")
    if mode == "top_k":
        top_k = kwargs.get("top_k", attn.shape[0])
    elif mode == "threshold":
        threshold = kwargs.get("threshold", 0.5)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Create a graph from the attention matrices
    G = nx.DiGraph()
    for i in range(attn.shape[0]):
        G.add_node(i)
    
    if mode == "top_k":
        for i in range(attn.shape[0]):
            top_indices = np.argsort(attn[i, :])[-top_k:]
            for j in top_indices:
                if i != j:
                    G.add_edge(i, j, weight=attn[i, j])
    elif mode == "threshold":
        for i in range(attn.shape[0]):
            for j in range(attn.shape[1]):
                if i != j and attn[i, j] > threshold:
                    G.add_edge(i, j, weight=attn[i, j])
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return G


class GraphFeatures:
    raised_interpolation_warning = False

    def __init__(self, attn_timestep_arr: np.ndarray, max_layers: int = None, analysis_type = "tokenwise", **kwargs):
        """
        Shape 4D: (layers, heads, n_query, n_key)
        """

        if not kwargs.get("prompt_attn", False):
            raise NotImplementedError("Only prompt analysis is supported for now")

        self.prompt_attn = kwargs.get("prompt_attn", False)
        self.top_k = kwargs.get("top_k", None) # choose top k edges
        self.analysis_type = analysis_type
        self.max_layers = max_layers
        #self.n_tokens = self.attn_arr.shape[1]

        self.feature_fn_map = {
            "clustering": lambda G: compute_average_clustering(G),
            "average_shortest_path_length": lambda G: compute_average_shortest_path_length(G),
            "forman_ricci": lambda G: compute_forman_ricci(G),
            "ollivier_ricci": lambda G: compute_ollivier_ricci(G),
            "average_degree": lambda G: compute_average_node_degree(G),
            "connectivity": lambda G: compute_connectivity(G),
            "sparseness": lambda G: compute_sparseness(G),
            "hubs": lambda G: compute_hubs(G),
            "clusters": lambda G: compute_communities(G),
            "communities": lambda G: compute_communities(G),
            "fourier": lambda G: compute_fourier_energy(G),
            #"cheeger_constant": self.extract_cheeger_constant,
            "commute_time_efficiency": lambda G: compute_efficiency(G),

            # New features
            "pagerank": lambda G: compute_max_pagerank(G),
            "eigenvector_centrality": lambda G: compute_max_eigenvector_centrality(G),
            "cycle_count": lambda G: compute_cycle_count(G),
        }

        self.raw_attn_matrices = attn_timestep_arr
        self.attn_graphs = self.create_graphs(self.raw_attn_matrices, **kwargs) # 1 graph per timestep


    def extract(self, feature_name, **kwargs):
        interpolate = kwargs.get("interpolate", False)
        if interpolate and self.analysis_type == "tokenwise" and not GraphFeatures.raised_interpolation_warning:
            print("WARNING: Interpolating tokenwise features is not supported, turning off interpolation.")
            interpolate = False
            GraphFeatures.raised_interpolation_warning = True

        feature_arr = []
        for attn_graph in self.attn_graphs:
            feature_arr.append(self.feature_fn_map[feature_name](attn_graph))
            
        if interpolate:
            feature_arr = self.__interpolate_to_max_layers(feature_arr)
        return feature_arr

    def create_layerwise_graphs(self, attn_arr, **kwargs):
        graphs = []

        attn_avg = np.mean(attn_arr, axis=1) # avg over heads

        for i in range(attn_avg.shape[0]):
            attn = attn_avg[i]
            graphs.append(create_graph_single_attn(attn, **kwargs))
        return graphs

    def create_tokenwise_graphs(self, attn_arr, **kwargs):
        token_wise_attns = []
        remove_sink = kwargs.get("remove_attention_sink", True)
        
        for i in range(len(attn_arr)): # for each token
            layer_attns = []
            for j in range(len(attn_arr[i])): # for each layer
                # we now have attn_arr[i][j] which is a 3D tensor of shape (1, n_heads, n_query, n_key)
                attn = (attn_arr[i][j]
                        .cpu() # move to cpu
                        .squeeze() # remove the 1-sized dimension
                        .mean(axis=0) # average over heads
                        .to(torch.float16) # convert to float16
                        .numpy()) # convert to numpy
                
                # Remove attention sink before aggregation if requested
                if remove_sink:
                    attn = attn[1:, 1:]  # Remove first row and column
                
                layer_attns.append(attn)
            
            aggregated_attn = aggregate_attention_layers(layer_attns)
            token_wise_attns.append(aggregated_attn)

        graphs = []
        for i in range(len(token_wise_attns)):
            attn = token_wise_attns[i]
            graphs.append(create_graph_single_attn(attn, **kwargs))
        return graphs
    

    def create_graphs(self, attn_arr, **kwargs):
        # FIXME !!!! Remove self.prompt_attn and fix the plot_raw_attention_matrices function to reduce dupe code
        """Create graphs from attention matrices with specified strategy."""
        if not self.prompt_attn:
            raise NotImplementedError("Intermediate attention not implemented, has diff shapes")
        
        # Create graphs based on analysis type
        if self.analysis_type == "layerwise":
            graphs = self.create_layerwise_graphs(attn_arr, **kwargs)
        elif self.analysis_type == "tokenwise":
            graphs = self.create_tokenwise_graphs(attn_arr, **kwargs)
        else:
            raise NotImplementedError("Invalid analysis type")
        
        return graphs
    

    def plot_attention_matrices(self, save_path, **kwargs):
        mode = kwargs.get("mode", "layerwise")
        if mode == "layerwise":
            self.plot_layerwise_attention_matrices(save_path)
        elif mode == "tokenwise":
            self.plot_tokenwise_attention_matrices(save_path)
        elif mode == "raw":
            self.plot_raw_attention_matrices(save_path)
        else:
            for i, attn_graph in enumerate(self.attn_graphs):
                plot_attention_matrix(nx.to_numpy_array(attn_graph), save_path + f"_{i}")


    def plot_tokenwise_attention_matrices(self, save_path):
        n = len(self.attn_graphs)
        # Determine grid dimensions (roughly square)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, attn_graph in enumerate(self.attn_graphs):
            ax = axes[i]
            attn_matrix = nx.to_numpy_array(attn_graph)
            sns.heatmap(attn_matrix, cmap="Reds", ax=ax, cbar=False)
            ax.set_title(f"Token {i}")
            ax.set_xlabel("Key Tokens")
            ax.set_ylabel("Query Tokens")
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(save_path + ".png", bbox_inches='tight')
        plt.close(fig)



    def plot_layerwise_attention_matrices(self, save_path):
        #if self.analysis_type != "layerwise":
        #    raise NotImplementedError("Only layerwise analysis is supported for now")
        
        n = len(self.attn_graphs)
        # Determine grid dimensions (roughly square)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
        # If only one subplot, wrap it in a list for consistency
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, attn_graph in enumerate(self.attn_graphs):
            ax = axes[i]
            attn_matrix = nx.to_numpy_array(attn_graph)
            sns.heatmap(attn_matrix, cmap="Reds", ax=ax, cbar=False)
            ax.set_title(f"Layer {i}")
            ax.set_xlabel("Key Tokens")
            ax.set_ylabel("Query Tokens")
        
        # Remove any unused subplots if n is not a perfect grid
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(save_path + ".png", bbox_inches='tight')
        plt.close(fig)

    def __interpolate_to_max_layers(self, feature_arr):
        if self.max_layers is None:
            return feature_arr
            
        curr_n_layers = len(feature_arr)
        new_n_layers = self.max_layers
        x_old = np.linspace(0, 1, curr_n_layers)
        x_new = np.linspace(0, 1, new_n_layers)
        return np.interp(x_new, x_old, feature_arr)

    def plot_raw_attention_matrices(self, save_path):
        attn_arr = self.raw_attn_matrices
        """Plot raw attention matrices before any processing."""
        if self.analysis_type == "layerwise":
            attn_avg = np.mean(attn_arr, axis=1)  # avg over heads
            matrices = attn_avg
            title_prefix = "Layer"
        else:  # tokenwise
            matrices = []
            for i in range(len(attn_arr)):
                layer_attns = []
                for j in range(len(attn_arr[i])):
                    attn = (attn_arr[i][j]
                            .cpu()
                            .squeeze()
                            .mean(axis=0)  # average over heads
                            .to(torch.float16)
                            .numpy())
                
                    # remove sink
                    attn = attn[1:, 1:]
                    layer_attns.append(attn)
                matrices.append(aggregate_attention_layers(layer_attns))
            title_prefix = "Token"
        
        n = len(matrices)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, matrix in enumerate(matrices):
            ax = axes[i]
            sns.heatmap(matrix, cmap="Reds", ax=ax, cbar=True)
            ax.set_title(f"Raw {title_prefix} {i} Attention")
            ax.set_xlabel("Key Tokens")
            ax.set_ylabel("Query Tokens")
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(save_path + "_raw.png", bbox_inches='tight')
        plt.close(fig)
