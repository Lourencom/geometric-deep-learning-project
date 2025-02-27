import os
import numpy as np
import torch
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
    def __init__(self, attn_timestep_arr: np.ndarray, max_layers: int, analysis_type = "layerwise", **kwargs):
        """
        attn_timestep_arr: np.ndarray -> for now its simply the prompt attentions
        Shape 4D: (layers, heads, n_query, n_key)
        """

        if not kwargs.get("prompt_attn", False):
            raise NotImplementedError("Only prompt analysis is supported for now")

        self.prompt_attn = kwargs.get("prompt_attn", False)
        self.top_k = kwargs.get("top_k", None) # choose top k edges
        self.attn_timestep_arr = attn_timestep_arr
        self.analysis_type = analysis_type
        self.max_layers = max_layers
        #self.n_tokens = self.attn_arr.shape[1]

        self.feature_fn_map = {
            "clustering": self.extract_average_clustering,
            "average_shortest_path_length": self.extract_average_shortest_path_length,
            "forman_ricci": self.extract_forman_ricci,
            "ollivier_ricci": self.extract_ollivier_ricci,
            "average_degree": self.extract_average_node_degree,
            "connectivity": self.extract_connectivity,
            "sparseness": self.extract_sparseness,
            "hubs": self.extract_hubs,
            "clusters": self.extract_clusters,
            "communities": self.extract_communities,
            "fourier": self.extract_fourier,
            #"cheeger_constant": self.extract_cheeger_constant,
            "commute_time_efficiency": self.extract_commute_time_efficiency,

            # New features
            "pagerank": self.extract_pagerank_centrality,
            "eigenvector_centrality": self.extract_eigenvector_centrality,
            "cycle_count": self.extract_cycle_count,
        }

        self.attn_graphs = self.create_graphs(attn_timestep_arr, **kwargs) # 1 graph per timestep

    def create_graphs(self, attn_arr, **kwargs):
        graphs = []
        if self.prompt_attn:
            if self.analysis_type != "tokenwise":
                attn_avg = np.mean(attn_arr, axis=1) # avg over heads
            else:
                attn_avg_over_heads = [
                    [
                        attn_arr[i][j].mean(axis=-3).squeeze().cpu().to(torch.float16).numpy()
                        for j in range(len(attn_arr[i]))
                    ]
                    for i in range(len(attn_arr))
                ]
                attn_avg = [aggregate_attention_layers(el) for el in attn_avg_over_heads]

            if self.analysis_type == "aggregated_layers":
                attn = aggregate_attention_layers(attn_avg) # aggregate over layers
                graphs.append(create_graph_single_attn(attn, **kwargs))
            elif self.analysis_type == "layerwise":
                for i in range(attn_avg.shape[0]):
                    attn = attn_avg[i]
                    graphs.append(create_graph_single_attn(attn, **kwargs))    
            elif self.analysis_type == "tokenwise":
                graphs = self.create_tokenwise_graphs(attn_avg, **kwargs)
            
        else:
            raise NotImplementedError("Intermediate attention not implemented, has diff shapes")

        return graphs

    def create_tokenwise_graphs(self, attn_arr, **kwargs):
        graphs = []

        for i in range(len(attn_arr)):
            attn = attn_arr[i]
            graphs.append(create_graph_single_attn(attn, **kwargs))
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
            node_degrees = [val for _, val in G.out_degree()]
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
    
    def extract_connectivity(self, **kwargs):
        """Return the number of strongly connected components for each directed graph."""
        connectivity = []
        for G in self.attn_graphs:
            comp = nx.number_strongly_connected_components(G)
            connectivity.append(comp)
        return np.array(connectivity)
    
    def extract_sparseness(self, **kwargs):
        """Compute sparseness as the inverse of density for each graph."""
        densities = []
        for G in self.attn_graphs:
            d = nx.density(G)
            densities.append(d)
        return np.array(densities)
    
    def extract_hubs(self, **kwargs):
        """Count the number of nodes with degree above the 90th percentile (as hubs)."""
        hubs_count = []
        for G in self.attn_graphs:
            degrees = np.array([deg for _, deg in G.degree()])
            if len(degrees) == 0:
                hubs_count.append(0)
                continue
            threshold = np.percentile(degrees, 90)
            count = np.sum(degrees >= threshold)
            hubs_count.append(count)
        return np.array(hubs_count)

    def extract_clusters(self, **kwargs):
        """Return the median clustering coefficient for each graph (using the undirected version)."""
        cluster_medians = []
        for G in self.attn_graphs:
            clustering_dict = nx.clustering(G.to_undirected(), weight="weight")
            med = np.median(list(clustering_dict.values())) if clustering_dict else 0
            cluster_medians.append(med)
        return np.array(cluster_medians)
    
    def extract_communities(self, **kwargs):
        """Detect communities using a greedy modularity algorithm and return the count per graph."""
        communities_count = []
        for G in self.attn_graphs:
            G_undir = G.to_undirected()
            communities = nx.algorithms.community.greedy_modularity_communities(G_undir, weight="weight")
            communities_count.append(len(communities))
        return np.array(communities_count)
    
    def extract_fourier(self, **kwargs):
        """
        Compute a summary statistic based on the graph Fourier transform.
        Here, we use the spectral energy: sum of absolute Laplacian eigenvalues divided by the number of nodes.
        """
        spectral_energy = []
        for G in self.attn_graphs:
            G_undir = G.to_undirected()
            n_nodes = G_undir.number_of_nodes()
            if n_nodes == 0:
                spectral_energy.append(0)
                continue
            L = nx.laplacian_matrix(G_undir).todense()
            eigvals = np.linalg.eigvals(L)
            energy = np.sum(np.abs(eigvals)) / n_nodes
            spectral_energy.append(energy)
        return np.array(spectral_energy)

    def extract_commute_time_efficiency(self, **kwargs):
        """
        Compute a surrogate metric for the efficiency of information propagation using the global efficiency.
        Note: global efficiency is defined for undirected graphs.
        """
        efficiency_vals = []
        for G in self.attn_graphs:
            G_undir = G.to_undirected()
            try:
                eff = nx.global_efficiency(G_undir)
            except Exception as e:
                eff = np.nan
            efficiency_vals.append(eff)
        return np.array(efficiency_vals)
    
    # New Feature: PageRank Centrality
    def extract_pagerank_centrality(self, aggregation = "max", **kwargs):
        pagerank= []
        for G in self.attn_graphs:
            # For directed graphs, nx.pagerank computes the PageRank.
            pr = nx.pagerank(G, weight='weight')
            pr_values = np.array(list(pr.values()))
            if aggregation == "mean":
                pagerank.append(np.mean(pr_values))
            elif aggregation == "max":
                pagerank.append(np.max(pr_values))
            else:
                raise ValueError(f"Invalid aggregation method: {aggregation}")
        return np.array(pagerank)
    
    # New Feature: Eigenvector Centrality
    def extract_eigenvector_centrality(self, aggregation = "max", **kwargs):
        eigencentrality = []
        for G in self.attn_graphs:
            try:
                ec = nx.eigenvector_centrality_numpy(G, weight='weight')
            except Exception:
                ec = {node: 0 for node in G.nodes()}
            ec_values = np.array(list(ec.values()))
            if aggregation == "mean":
                eigencentrality.append(np.mean(ec_values))
            elif aggregation == "max":
                eigencentrality.append(np.max(ec_values))
            else:
                raise ValueError(f"Invalid aggregation method: {aggregation}")
        return np.array(eigencentrality)
    
    # New Feature: Cycle Count (using cycle basis in undirected graph as a proxy for loops)
    def extract_cycle_count(self, **kwargs):
        cycle_counts = []
        for G in self.attn_graphs:
            G_undir = G.to_undirected()
            # nx.cycle_basis returns a list of cycles (as lists of nodes)
            cycles = nx.cycle_basis(G_undir)
            cycle_counts.append(len(cycles))
        return np.array(cycle_counts)
    

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
    stored_prompt_attns = load_attns(args, models=args.models, attn_dir=args.attn_dir, save=True, tokenwise=args.analysis_type == "tokenwise")
    
    # Create graph features for each model
    graph_features = {}
    for i, model_tuple in enumerate(args.models):
        family, size, variant = model_tuple
        model_identifier = f"{family}_{size}_{variant}"
        
        analysis_type = args.analysis_type
        graph_features[model_identifier] = GraphFeatures(
            stored_prompt_attns[i], 
            prompt_attn=True, 
            remove_attention_sink=True,
            max_layers=32,
            analysis_type=analysis_type
        )

        if args.plot_matrices:
            # Plot attention matrices
            matrix_filename = os.path.join(args.output_dir, f"{base_filename}_{model_identifier}_no_sink_layerwise")
            graph_features[model_identifier].plot_layerwise_attention_matrices(matrix_filename)

    features = [
        'clustering', 
        'average_shortest_path_length', 
        'forman_ricci',
        #'ollivier_ricci',
        'average_degree',
        'connectivity',
        'sparseness',
        'hubs',
        'clusters',
        'communities',
        #'fourier',
        # 'cheeger_constant',
        'commute_time_efficiency',
        # new
        'pagerank',
        'eigenvector_centrality',
        'cycle_count',
    ]

    # Create a single figure with subplots for all features
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + 1) // 2  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Plot each feature in its own subplot
    linestyles = ['-', '--', ':', '-.'] # Different line styles
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p'] # Different markers
    
    for idx, feature in enumerate(features):
        print(f"Running {feature}...")
        ax = axes[idx]
        
        for i, model_tuple in enumerate(args.models):
            family, size, variant = model_tuple
            model_identifier = f"{family}_{size}_{variant}"
            
            # Cycle through linestyles and markers
            linestyle = linestyles[i % len(linestyles)]
            marker = markers[i % len(markers)]

            # Handle features that return multiple values
            if feature in ['pagerank', 'eigenvector_centrality']:
                # mean_values = graph_features[model_identifier].extract(feature, interpolate=True, aggregation="mean")
                max_values = graph_features[model_identifier].extract(feature, interpolate=True, aggregation="max")
                """
                ax.plot(mean_values, 
                       label=f"{model_identifier} (mean)",
                       linestyle=linestyle,
                       marker=marker,
                       markevery=5,
                       markersize=4)
                """
                ax.plot(max_values, 
                       label=f"{model_identifier} (max)",
                       linestyle=linestyle,
                       marker=marker,
                       markevery=5,
                       markersize=4,
                       alpha=0.5)  # Make max values slightly transparent
            else:
                feature_values = graph_features[model_identifier].extract(feature, interpolate=True)
                ax.plot(feature_values, 
                       label=model_identifier,
                       linestyle=linestyle,
                       marker=marker,
                       markevery=5,
                       markersize=4)
            
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
    savefig_path = os.path.join(args.output_dir, f"{base_filename}_all_features_no_sink_{args.analysis_type}_{append_str}")
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
