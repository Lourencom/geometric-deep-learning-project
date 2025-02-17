import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from GraphRicciCurvature.FormanRicci import FormanRicci


from attention import extract_attention
from model import run_model
from visualization import plot_attention_matrices

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
            "forman_ricci": self.extract_forman_ricci
        }
        self.max_layers = max_layers
        self.threshold = threshold

        self.create_graphs(self.attn_arr, self.threshold)

    def create_graphs(self, attn_arr, threshold):
        self.graphs = [self.__create_graph_single_attn(attn, threshold) for attn in attn_arr]

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
    
    def extract_average_clustering(self):
        return np.array([nx.average_clustering(G) for G in self.graphs])
    
    def extract_average_shortest_path_length(self):
        return np.array([nx.average_shortest_path_length(G) for G in self.graphs])
    
    def extract_forman_ricci(self):
        frc = [FormanRicci(G) for G in self.graphs]
        for i in range(len(frc)):
            frc[i].compute_ricci_curvature()

        edges = [frc[i].G.edges(data=True) for i in range(len(frc))]
        curvature_hists = [
            np.median([data['formanCurvature'] for _,_,data in e])
            for e in edges
        ]
        return np.array(curvature_hists)
    
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
    cached_attentions = [el for el in os.listdir(args.output_dir) if "attention_values" in el and el.endswith(".npy")]
    cached_attentions = [el for el in cached_attentions if model_size in el]
    return cached_attentions

def load_attns(args, model_sizes=["large", "small"]):
    attn_arrs = []
    for model_size in model_sizes:
        cached_attentions = get_cached_attention(args, model_size)
        if len(cached_attentions) == 0:
            args.model_size = model_size
            outputs, _, _ = run_model(args)
            attn_arr = extract_attention(args, outputs)
            np.save(args.output_dir + "/" + f"attention_values_{model_size}_{args.prompt_difficulty}_{args.prompt_category}_{args.prompt_n_shots}.npy", attn_arr)
        else:
            attn_arr = np.load(args.output_dir + "/" + cached_attentions[0])
        attn_arrs.append(attn_arr)
    return attn_arrs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, default="data/prompts.json")
    parser.add_argument("--prompt_difficulty", type=str, default="medium")
    parser.add_argument("--prompt_category", type=str, default=None)
    parser.add_argument("--prompt_n_shots", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="data/attn")
    
    args = parser.parse_args()

    attn_arrs = load_attns(args)
    attn_arr_large = attn_arrs[0]
    attn_arr_small = attn_arrs[1]


    #features=['clustering', 'average_shortest_path_length', 'forman_ricci']
    features = ['forman_ricci']
    
    graph_features_large = GraphFeatures(attn_arr_large)
    graph_features_small = GraphFeatures(attn_arr_small)

    # Create a single figure for all thresholds
    fig = plt.figure(figsize=(12, 8))
    #thresholds = [0.01, 0.02, 0.05, 0.1]
    threshold = 0.02

    for feature in features:
        feature_large = graph_features_large.extract(feature, threshold=threshold, interpolate=True)
        feature_small = graph_features_small.extract(feature, threshold=threshold, interpolate=True)
        
        # Plot with threshold-specific labels
        plt.plot(feature_large, label=f"large (t={threshold})", linestyle='-')
        plt.plot(feature_small, label=f"small (t={threshold})", linestyle='--')

    plt.xlabel("N layers")
    plt.ylabel(feature)
    plt.title(f"{feature} for large and small models on a {args.prompt_difficulty} prompt")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output_dir + f"/{feature}.png", bbox_inches='tight')
    plt.show()
