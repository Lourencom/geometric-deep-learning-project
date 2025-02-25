import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from utils import filter_prompts
from attention import extract_attention
from model import run_model
from visualization import plot_attention_matrix
from utils import relative_to_absolute_path
from args import get_args

def aggregate_attention_layers(attn_matrices):
    """
    Aggregates a list of attention matrices using matrix multiplication.
    
    Parameters:
        attn_matrices (list of np.ndarray): 
            List of attention matrices where each matrix is of shape (n_tokens, n_tokens).
            The list should be ordered from the first layer to the last layer.
            
    Returns:
        np.ndarray: Aggregated attention matrix of shape (n_tokens, n_tokens) that 
                    represents the overall information flow across layers.
                    
    Example:
        For two layers, the aggregated attention is computed as:
            A_agg = A_layer2 @ A_layer1
    """
    # Start with the first layer's attention matrix.
    A_agg = attn_matrices[0]
    # Multiply successively by the next layer's attention matrix.
    for attn in attn_matrices[1:]:
        A_agg = np.dot(attn, A_agg)
    return A_agg

class GraphFeatures:
    def __init__(self, attn_arr: np.ndarray, max_layers: int=32, threshold: float=0.1):
        """
        attn_arr: np.ndarray
            The attention array of shape (batch_size, n_heads, n_tokens, n_tokens)
        """
        self.attn_arr = np.mean(attn_arr, axis=1) # average over heads
        self.n_tokens = self.attn_arr.shape[1]

        self.feature_fn_map = {
            "clustering": self.extract_average_clustering,
            "average_shortest_path_length": self.extract_average_shortest_path_length,
            "forman_ricci": self.extract_forman_ricci,
            "ollivier_ricci": self.extract_ollivier_ricci,
            "average_degree": self.extract_average_node_degree,
        }
        self.max_layers = max_layers
        self.threshold = threshold

        self.create_graphs(self.attn_arr, self.threshold)

    def create_graphs(self, attn_arr, threshold):
        aggregated_attn = aggregate_attention_layers(attn_arr)
        self.graphs = [self.__create_graph_single_attn(aggregated_attn, threshold)]

    def __create_graph_single_attn(self, attn, threshold):
        """
        I am given a single attention matrix (NxN) and I want to create a graph from it.
        Edges are present if the attention weight between the ith and jth tokens is greater than the threshold.
        """
        # Create a graph from the attention matrices
        G = nx.Graph()

        #th_quantile = np.quantile(attn, threshold)
        # Add nodes to the graph
        for i in range(attn.shape[0]):
            G.add_node(i)
        
        # Add edges to the graph
        for i in range(attn.shape[0]):
            for j in range(attn.shape[1]):
                if i != j and attn[i, j] > threshold:
                    G.add_edge(i, j)
        
        return G
    
    def extract_average_node_degree(self):
        avg_degrees = []
        for G in self.graphs:
            node_degrees = G.degree()
            avg_degree = np.mean(node_degrees)
            avg_degrees.append(avg_degree)
        return np.array(avg_degrees)

    def extract_average_clustering(self):
        return np.array([nx.average_clustering(G) for G in self.graphs])
    
    def extract_average_shortest_path_length(self):
        return np.array([nx.average_shortest_path_length(G) for G in self.graphs])
    
    def extract_ollivier_ricci(self):
        self.create_graphs(self.attn_arr, 0)
        orc = [OllivierRicci(G) for G in self.graphs]
        for i in range(len(orc)):
            orc[i].compute_ricci_curvature()

        edges = [orc[i].G.edges(data=True) for i in range(len(orc))]
        medians = [
            np.median([data['ricciCurvature'] for _,_,data in e])
            for e in edges
        ]
        return np.array(medians)
    

    def extract_forman_ricci(self):
        frc = [FormanRicci(G) for G in self.graphs]
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
            self.create_graphs(self.attn_arr, threshold)

        feature_arr = self.feature_fn_map[feature_name]()
        if interpolate:
            feature_arr = self.__interpolate_to_max_layers(feature_arr)
        return feature_arr
    
def get_cached_attention(args, model_size):
    cached_attentions = [el for el in os.listdir(relative_to_absolute_path(args.attn_dir)) if "attention_values" in el and el.endswith(".npy")]
    
    cached_attentions = filter_prompts(cached_attentions, args.prompt_difficulty, args.prompt_category, args.prompt_n_shots, model_size)
    
    return cached_attentions

def load_attns(args, model_sizes=["large", "small"]):
    attn_arrs = []
    for model_size in model_sizes:
        cached_attentions = get_cached_attention(args, model_size)
        if len(cached_attentions) == 0:
            args.model_size = model_size
            outputs, _, _ = run_model(args)
            attn_arr = extract_attention(args, outputs)
            np.save(relative_to_absolute_path(args.output_dir) + "/" + f"attention_values_{model_size}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.npy", attn_arr)
        else:
            attn_arr = np.load(relative_to_absolute_path(args.attn_dir) + "/" + cached_attentions[0])
        attn_arrs.append(attn_arr)
    return attn_arrs


if __name__ == "__main__":
    args = get_args()
    args.output_dir = "media/feature_plots"  # Override default for graph features

    attn_dicts = load_attns(args)

    attn_arrs = load_attns(args)
    attn_arr_large = attn_arrs[0]
    attn_arr_small = attn_arrs[1]


    features=['clustering', 'average_shortest_path_length', 
              'forman_ricci', 'ollivier_ricci', 'average_degree']
    
    graph_features_large = GraphFeatures(attn_arr_large)
    graph_features_small = GraphFeatures(attn_arr_small)


    os.makedirs(relative_to_absolute_path(args.output_dir), exist_ok=True)

    # Create a single figure for all thresholds
    fig = plt.figure(figsize=(12, 8))
    #thresholds = [0.01, 0.02, 0.05, 0.1]
    threshold = 0.01

    #breakpoint()

    for feature in features:
        feature_large = graph_features_large.extract(feature, threshold=threshold, interpolate=True)
        feature_small = graph_features_small.extract(feature, threshold=threshold, interpolate=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot with threshold-specific labels
        ax.plot(feature_large, label=f"large (t={threshold})", linestyle='-')
        ax.plot(feature_small, label=f"small (t={threshold})", linestyle='--')

        ax.set_xlabel("N layers")
        ax.set_ylabel(feature)
        ax.set_title(f"{feature} for large and small models on a {args.prompt_difficulty} prompt")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(relative_to_absolute_path(args.output_dir) + f"/{feature}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.png", bbox_inches='tight')
        plt.show()
        plt.close(fig)
