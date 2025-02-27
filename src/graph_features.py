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

    Selects top k outgoing edges (query -> keys) for each query node.
    """
    top_k = kwargs.get("top_k", attn.shape[0])

    # Create a graph from the attention matrices
    G = nx.DiGraph()

    # Add nodes to the graph
    for i in range(attn.shape[0]):
        G.add_node(i)
    
    # Add edges to the graph
    for i in range(attn.shape[0]):
        top_indices = np.argsort(attn[i, :])[-top_k:]
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
        return np.array([nx.average_clustering(G, weight="weight") for G in self.attn_graphs])
    
    def extract_average_shortest_path_length(self, **kwargs):
        return np.array([nx.average_shortest_path_length(G, weight="weight") for G in self.attn_graphs])
    
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


def analyze_prompt(args, prompt_id):
    """Run the analysis for a single prompt"""
    args.prompt_id = prompt_id  # Set current prompt ID
    print(f"\nAnalyzing prompt {prompt_id}...")

    base_filename = f"prompt_{prompt_id}"
    
    # Load attention data
    stored_prompt_attns = load_attns(args, models=args.models, attn_dir=args.attn_dir, save=True)
    
    # Create graph features for each model
    graph_features = {}
    for i, model_tuple in enumerate(args.models):
        family, size, variant = model_tuple
        model_identifier = f"{family}_{size}_{variant}"
        
        graph_features[model_identifier] = GraphFeatures(stored_prompt_attns[i], 
                                                       prompt_attn=True, 
                                                       remove_attention_sink=True)
        
        # Plot attention matrices
        matrix_filename = os.path.join(args.output_dir, f"{base_filename}_{model_identifier}_no_sink_layerwise")
        graph_features[model_identifier].plot_layerwise_attention_matrices(matrix_filename)

    features = [
        'clustering', 
        'average_shortest_path_length', 
        'forman_ricci',
        #'ollivier_ricci',
        'average_degree'
    ]

    # Create a single figure with subplots for all features
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Plot each feature in its own subplot
    for idx, feature in enumerate(features):
        print(f"Running {feature}...")
        ax = axes[idx]
        
        for model_tuple in args.models:
            family, size, variant = model_tuple
            model_identifier = f"{family}_{size}_{variant}"
            
            feature_values = graph_features[model_identifier].extract(feature, interpolate=True, max_layers=32)
            ax.plot(feature_values, label=model_identifier)
            
        ax.set_xlabel("Layers")
        ax.set_ylabel(feature)
        ax.set_title(feature)
        ax.grid(True)
    
    # Remove any unused subplots
    for idx in range(len(features), len(axes)):
        fig.delaxes(axes[idx])
    
    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    # Adjust layout and save
    fig.suptitle(f"Graph Features Analysis for Prompt {prompt_id}", y=1.02)
    plt.tight_layout()
    append_str = "_".join([f"{family}_{size}_{variant}" for family, size, variant in args.models])
    savefig_path = os.path.join(args.output_dir, f"{base_filename}_all_features_no_sink_layerwise_{append_str}")
    plt.savefig(savefig_path + ".png", bbox_inches='tight')
    plt.close(fig)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.attn_dir, exist_ok=True)

    for prompt_id in args.prompt_ids:
        analyze_prompt(args, prompt_id)

    print("\nDone")


if __name__ == "__main__":
    main()
