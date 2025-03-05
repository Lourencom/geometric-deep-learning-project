"""Module containing graph metric computations"""
import networkx as nx
import numpy as np
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

def compute_average_node_degree(G):
    """Compute average out-degree of nodes in the graph."""
    node_degrees = [val for _, val in G.out_degree()]
    return np.mean(node_degrees) if node_degrees else 0

def compute_average_clustering(G):
    """Compute average clustering coefficient."""
    return nx.average_clustering(G, weight="weight")

def compute_average_shortest_path_length(G):
    """Compute average shortest path length."""
    try:
        return nx.average_shortest_path_length(G, weight="weight")
    except nx.NetworkXError:
        return float('inf')  # Return inf for disconnected graphs

def compute_ollivier_ricci(G):
    """Compute median Ollivier-Ricci curvature."""
    orc = OllivierRicci(G)
    orc.compute_ricci_curvature()
    edges = orc.G.edges(data=True)
    return np.median([data['ricciCurvature'] for _,_,data in edges]) if edges else 0

def compute_forman_ricci(G):
    """Compute median Forman-Ricci curvature."""
    frc = FormanRicci(G)
    frc.compute_ricci_curvature()
    edges = frc.G.edges(data=True)
    return np.median([data['formanCurvature'] for _,_,data in edges]) if edges else 0

def compute_connectivity(G):
    """Compute number of strongly connected components."""
    return nx.number_strongly_connected_components(G)

def compute_sparseness(G):
    """Compute graph density."""
    return nx.density(G)

def compute_hubs(G, percentile=90):
    """Count nodes with degree above the specified percentile."""
    degrees = np.array([deg for _, deg in G.degree()])
    if len(degrees) == 0:
        return 0
    threshold = np.percentile(degrees, percentile)
    return np.sum(degrees >= threshold)

def compute_communities(G):
    """Count number of communities using greedy modularity."""
    G_undir = G.to_undirected()
    communities = nx.algorithms.community.greedy_modularity_communities(G_undir, weight="weight")
    return len(communities)

def compute_fourier_energy(G):
    """Compute spectral energy from graph Laplacian."""
    G_undir = G.to_undirected()
    n_nodes = G_undir.number_of_nodes()
    if n_nodes == 0:
        return 0
    L = nx.laplacian_matrix(G_undir).todense()
    eigvals = np.linalg.eigvals(L)
    return np.sum(np.abs(eigvals)) / n_nodes

def compute_efficiency(G):
    """Compute global efficiency."""
    G_undir = G.to_undirected()
    try:
        return nx.global_efficiency(G_undir)
    except Exception:
        return np.nan

def compute_max_pagerank(G):
    """Compute maximum PageRank value."""
    try:
        pr = nx.pagerank(G, weight='weight')
        return np.max(list(pr.values())) if pr else 0
    except Exception:
        return 0

def compute_max_eigenvector_centrality(G):
    """Compute maximum eigenvector centrality."""
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight='weight')
        return np.max(list(ec.values())) if ec else 0
    except Exception:
        return 0

def compute_cycle_count(G):
    """Count number of cycles in undirected version of graph."""
    G_undir = G.to_undirected()
    return len(nx.cycle_basis(G_undir))

def compute_attention_fidelity(G):
    """Compute attention fidelity."""
    return 0

def compute_fidelity_mixing_tradeoff(G):
    """Compute fidelity mixing tradeoff."""
    return 0
