import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from utils import filter_prompts
from attention import extract_attention, aggregate_attention_layers
from model import run_model
from visualization import plot_attention_matrix
from utils import relative_to_absolute_path
from args import get_args


def create_graph_single_attn(attn, **kwargs):
    """
    I am given a single attention matrix (NxN) and I want to create a graph from it.
    Edges are chosen based on the top k values in the attention matrix.
    """
    top_k = kwargs.get("top_k", attn.shape[0])

    # Create a graph from the attention matrices
    G = nx.Graph()

    #th_quantile = np.quantile(attn, threshold)
    # Add nodes to the graph
    for i in range(attn.shape[0]):
        G.add_node(i)
    
    # Add edges to the graph
    for i in range(attn.shape[0]):
        top_indices = np.argsort(attn[i, :])[:top_k]
        for j in top_indices:
            if i != j: # FIXME: we are removing self-attention, why pavle?
                G.add_edge(i, j)
    return G


class GraphFeatures:
    def __init__(self, attn_timestep_arr: np.ndarray, max_layers: int=32, analysis_type = "layerwise", **kwargs):
        """
        attn_timestep_arr: np.ndarray
            The attention array of shape (timesteps, layers, batch_size, n_heads, n_tokens, n_tokens)
        """

        if analysis_type != "layerwise":
            raise NotImplementedError("Only layerwise analysis is supported for now")
        if not kwargs.get("prompt_attn", False):
            raise NotImplementedError("Only prompt analysis is supported for now")

        self.max_layers = max_layers
        self.threshold = kwargs.get("threshold", 0.0)
        self.prompt_attn = kwargs.get("prompt_attn", False)
        self.top_k = kwargs.get("top_k", None) # choose top k edges
        self.attn_timestep_arr = attn_timestep_arr
        #self.n_tokens = self.attn_arr.shape[1]

        self.feature_fn_map = {
            "clustering": self.extract_average_clustering,
            "average_shortest_path_length": self.extract_average_shortest_path_length,
            "forman_ricci": self.extract_forman_ricci,
            "ollivier_ricci": self.extract_ollivier_ricci,
            "average_degree": self.extract_average_node_degree,
        }

        self.attn_graphs = self.create_graphs(attn_timestep_arr, **kwargs) # 1 graph per timestep

    def create_graphs(self, attn_arr, **kwargs):
        graphs = []

        if not self.prompt_attn:
            raise NotImplementedError("Intermediate attention not implemented, has diff shapes")

        for attn in attn_arr:
            attn_avg = np.mean(attn, axis=1) # avg over heads
            aggregated_attn = aggregate_attention_layers(attn_avg) # aggregate over layers
            graphs.append(create_graph_single_attn(aggregated_attn, **kwargs))    
            
        return graphs

    def plot_attention_matrices(self, save_path):
        for attn_graph in self.attn_graphs:
            plot_attention_matrix(nx.to_numpy_array(attn_graph), save_path)
    
    def extract_average_node_degree(self):
        avg_degrees = []
        for G in self.attn_graphs:
            node_degrees = G.degree()
            avg_degree = np.mean(node_degrees)
            avg_degrees.append(avg_degree)
        return np.array(avg_degrees)

    def extract_average_clustering(self):
        return np.array([nx.average_clustering(G) for G in self.attn_graphs])
    
    def extract_average_shortest_path_length(self):
        return np.array([nx.average_shortest_path_length(G) for G in self.attn_graphs])
    
    def extract_ollivier_ricci(self):
        orc = [OllivierRicci(G) for G in self.attn_graphs]
        for i in range(len(orc)):
            orc[i].compute_ricci_curvature()

        edges = [orc[i].G.edges(data=True) for i in range(len(orc))]
        medians = [
            np.median([data['ricciCurvature'] for _,_,data in e])
            for e in edges
        ]
        return np.array(medians)
    

    def extract_forman_ricci(self):
        frc = [FormanRicci(G) for G in self.attn_graphs]
        for i in range(len(frc)):
            frc[i].compute_ricci_curvature()

        edges = [frc[i].G.edges(data=True) for i in range(len(frc))]
        medians = [
            np.median([data['formanCurvature'] for _,_,data in e])
            for e in edges
        ]
        return np.array(medians)
    
    def __interpolate_to_max_layers(self, feature_arr):
        curr_n_layers = len(feature_arr)
        new_n_layers = self.max_layers
        x_old = np.linspace(0, 1, curr_n_layers)
        x_new = np.linspace(0, 1, new_n_layers)
        return np.interp(x_new, x_old, feature_arr)
    
    def extract(self, feature_name, interpolate=False, threshold=None):
        if threshold and threshold != self.threshold: # we can override the threshold and recreate the graphs
            self.threshold = threshold
            self.attn_graphs = self.create_graphs(self.attn_timestep_arr, threshold=threshold)

        feature_arr = self.feature_fn_map[feature_name]()
        if interpolate:
            feature_arr = self.__interpolate_to_max_layers(feature_arr)
        return feature_arr
    
def get_cached_attention(args, model_size):
    cached_attention_files = [el for el in os.listdir(relative_to_absolute_path(args.attn_dir))
                        if "attention_values" in el and el.endswith(".npy")]
    
    cached_attentions = filter_prompts(cached_attention_files, args.prompt_difficulty, args.prompt_category, args.prompt_n_shots, model_size)
    
    return cached_attentions

def load_attns(args, model_sizes=["small", "large"]):
    attn_dicts = []
    for model_size in model_sizes:
        cached_attentions = get_cached_attention(args, model_size)
        attn_path = relative_to_absolute_path(args.attn_dir)
        if len(cached_attentions) == 0:
            args.model_size = model_size
            outputs, *_ = run_model(args)
            attn_dict = extract_attention(args, outputs, save=False)
        else:
            loaded = np.load(os.path.join(attn_path, cached_attentions[0]))
            attn_dict = loaded.item() if isinstance(loaded, np.ndarray) else loaded
        attn_dicts.append(attn_dict)
    return attn_dicts


if __name__ == "__main__":
    args = get_args()
    args.output_dir = "media/feature_plots"  # Override default for graph features

    attn_dicts = load_attns(args)

    breakpoint()

    attn_arr_prompt_small = attn_dicts[0]["prompt_attns"]
    attn_arr_prompt_large = attn_dicts[1]["prompt_attns"]

    attn_arr_intermediate_small = attn_dicts[0]["intermediate_attns"]
    attn_arr_intermediate_large = attn_dicts[1]["intermediate_attns"]

    # intermediate attns also have beam dimension, I need to see how to handle that


    features=['clustering', 'average_shortest_path_length', 
              'forman_ricci', 'ollivier_ricci', 'average_degree']
    
    graph_features_small = GraphFeatures(attn_arr_prompt_small, prompt_attn=True)
    graph_features_large = GraphFeatures(attn_arr_prompt_large, prompt_attn=True)

    # graph_features_intermediate_small = GraphFeatures(attn_arr_intermediate_small, prompt_attn=False)
    # graph_features_intermediate_large = GraphFeatures(attn_arr_intermediate_large, prompt_attn=False)

    os.makedirs(relative_to_absolute_path(args.output_dir), exist_ok=True)

    # Create a single figure for all thresholds
    fig = plt.figure(figsize=(12, 8))
    #thresholds = [0.01, 0.02, 0.05, 0.1]
    threshold = 0.01


    for feature in features:
        feature_large = graph_features_large.extract(feature, threshold=threshold, interpolate=True)
        feature_small = graph_features_small.extract(feature, threshold=threshold, interpolate=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot with threshold-specific labels
        ax.plot(feature_large, label=f"large (t={threshold})", linestyle='-')
        ax.plot(feature_small, label=f"small (t={threshold})", linestyle='--')

        ax.set_xlabel("timesteps")
        ax.set_ylabel(feature)
        ax.set_title(f"{feature} for large and small models on a {args.prompt_difficulty} prompt")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        plt.tight_layout()
        savefig_path = relative_to_absolute_path(args.output_dir) + f"/prompt_{feature}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.png"
        plt.savefig(savefig_path, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    graph_features_small.plot_attention_matrices(relative_to_absolute_path(args.output_dir) + f"/prompt_attention_matrices_small_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.png")
    graph_features_large.plot_attention_matrices(relative_to_absolute_path(args.output_dir) + f"/prompt_attention_matrices_large_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.png")
