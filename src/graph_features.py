import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from attention import aggregate_attention_layers, load_attns
from visualization import plot_attention_matrix
from utils import relative_to_absolute_path
from args import get_args
import math
import seaborn as sns


def create_graph_single_attn(attn, **kwargs):
    """
    I am given a single attention matrix (NxN) and I want to create a graph from it.
    Edges are chosen based on the top k values in the attention matrix.
    """
    top_k = kwargs.get("top_k", attn.shape[0])

    # Create a graph from the attention matrices
    G = nx.DiGraph()

    # Add nodes to the graph
    for i in range(attn.shape[0]):
        G.add_node(i)
    
    # Add edges to the graph
    for i in range(attn.shape[0]):
        top_indices = np.argsort(attn[i, :])[:top_k]
        for j in top_indices:
            if i != j: # FIXME: we are removing self-attention, why pavle?
                G.add_edge(i, j, weight=attn[i, j])
    
    if kwargs.get("remove_attention_sink", False): # remove first column
        G.remove_node(0)
    return G


class GraphFeatures:
    def __init__(self, attn_timestep_arr: np.ndarray, max_layers: int=32, analysis_type = "layerwise", **kwargs):
        """
        attn_timestep_arr: np.ndarray -> for now its simply the prompt attentions
        Shape 4D: (layers, heads, n_query, n_key)
        """

        if analysis_type != "layerwise":
            raise NotImplementedError("Only layerwise analysis is supported for now")
        if not kwargs.get("prompt_attn", False):
            raise NotImplementedError("Only prompt analysis is supported for now")

        self.max_layers = max_layers
        self.prompt_attn = kwargs.get("prompt_attn", False)
        self.top_k = kwargs.get("top_k", None) # choose top k edges
        self.attn_timestep_arr = attn_timestep_arr
        self.analysis_type = analysis_type
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
        if self.prompt_attn:
            attn_avg = np.mean(attn_arr, axis=1) # avg over heads

            if self.analysis_type == "aggregated_layers":
                attn = aggregate_attention_layers(attn_avg) # aggregate over layers
                graphs.append(create_graph_single_attn(attn, **kwargs))
            elif self.analysis_type == "layerwise":
                for i in range(attn_avg.shape[0]):
                    attn = attn_avg[i]
                    graphs.append(create_graph_single_attn(attn, **kwargs))    
            else:
                raise NotImplementedError("Invalid analysis type")
            
        else:
            raise NotImplementedError("Intermediate attention not implemented, has diff shapes")

        return graphs

    def plot_layerwise_attention_matrices(self, save_path):
        if self.analysis_type != "layerwise":
            raise NotImplementedError("Only layerwise analysis is supported for now")
        
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


    def plot_attention_matrices(self, save_path):
        for i, attn_graph in enumerate(self.attn_graphs):
            plot_attention_matrix(nx.to_numpy_array(attn_graph), save_path + f"_{i}")
    
    def extract_average_node_degree(self, **kwargs):
        avg_degrees = []
        for G in self.attn_graphs:
            node_degrees = G.degree()
            avg_degree = np.mean(node_degrees)
            avg_degrees.append(avg_degree)
        return np.array(avg_degrees)

    def extract_average_clustering(self, **kwargs):
        return np.array([nx.average_clustering(G) for G in self.attn_graphs])
    
    def extract_average_shortest_path_length(self, **kwargs):
        return np.array([nx.average_shortest_path_length(G) for G in self.attn_graphs])
    
    def extract_ollivier_ricci(self, **kwargs):
        orc = [OllivierRicci(G) for G in self.attn_graphs]
        for i in range(len(orc)):
            orc[i].compute_ricci_curvature()

        edges = [orc[i].G.edges(data=True) for i in range(len(orc))]
        medians = [
            np.median([data['ricciCurvature'] for _,_,data in e])
            for e in edges
        ]
        return np.array(medians)
    

    def extract_forman_ricci(self, **kwargs):
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
    
    def extract(self, feature_name, **kwargs):
        interpolate = kwargs.get("interpolate", False)
        top_k = kwargs.get("top_k", self.top_k)
        
        if top_k != self.top_k: # recreate graphs with new top_k
            self.top_k = top_k
            self.attn_graphs = self.create_graphs(self.attn_timestep_arr, top_k=top_k)

        feature_arr = self.feature_fn_map[feature_name](**kwargs)
        if interpolate:
            feature_arr = self.__interpolate_to_max_layers(feature_arr)
        return feature_arr


if __name__ == "__main__":
    args = get_args()
    args.output_dir = "media/feature_plots"  # Override default for graph features
    os.makedirs(relative_to_absolute_path(args.output_dir), exist_ok=True)

    attn_dicts = load_attns(args)

    #breakpoint()

    small_attns, large_attns = attn_dicts

    attn_arr_prompt_small = small_attns["prompt_attns"]
    #attn_arr_intermediate_small = small_attns["intermediate_attns"]
    
    attn_arr_prompt_large = large_attns["prompt_attns"]
    #attn_arr_intermediate_large = large_attns["intermediate_attns"]

    features=['clustering', 
              'average_shortest_path_length', 
              'forman_ricci',
              #'ollivier_ricci',
              'average_degree'
              ]
    
    graph_features_small = GraphFeatures(attn_arr_prompt_small, prompt_attn=True, remove_attention_sink=True)
    graph_features_large = GraphFeatures(attn_arr_prompt_large, prompt_attn=True, remove_attention_sink=True)

    # graph_features_intermediate_small = GraphFeatures(attn_arr_intermediate_small, prompt_attn=False)
    # graph_features_intermediate_large = GraphFeatures(attn_arr_intermediate_large, prompt_attn=False)

    graph_features_small.plot_layerwise_attention_matrices(relative_to_absolute_path(args.output_dir) + f"/prompt_attention_matrices_small_no_sink_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}")
    graph_features_large.plot_layerwise_attention_matrices(relative_to_absolute_path(args.output_dir) + f"/prompt_attention_matrices_large_no_sink_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}")
   
    fig = plt.figure(figsize=(12, 8))

    for feature in features:
        print(feature)
        feature_large = graph_features_large.extract(feature, interpolate=True, max_layers=32)
        feature_small = graph_features_small.extract(feature, interpolate=True, max_layers=32)
        
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(feature_large, label=f"large", linestyle='-')
        ax.plot(feature_small, label=f"small", linestyle='--')

        ax.set_xlabel("timesteps")
        ax.set_ylabel(feature)
        ax.set_title(f"{feature} for large and small models on a {args.prompt_difficulty} prompt")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        plt.tight_layout()
        savefig_path = relative_to_absolute_path(args.output_dir) + f"/prompt_{feature}_no_sink_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}"
        plt.savefig(savefig_path + ".png", bbox_inches='tight')
        plt.show()
        plt.close(fig)

    print("Done")