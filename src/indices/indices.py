import networkx as nx
import numpy as np
from collections import Counter
from math import log


def balaban_j_index(G):
    """
    Calculates the Balaban J index of a NetworkX graph G.
    """
    n = G.number_of_nodes()
    c = nx.degree_centrality(G)
    s = 0
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                s += c[i] * c[j] / (1 + nx.shortest_path_length(G, i, j))
    return s


def hosoya_z_index(G):
    n = nx.number_of_nodes(G)
    A = nx.adjacency_matrix(G).todense()
    A_squared = A @ A
    diag = [A_squared[i, i] for i in range(n)]
    return sum(diag) / (n * (n - 1))


def chromatic_information_index(G):
    colors = nx.greedy_color(G)
    color_counts = Counter(colors.values())
    total_nodes = nx.number_of_nodes(G)
    entropy = -sum(count/total_nodes * log(count/total_nodes, 2)
                   for count in color_counts.values())
    return entropy


def szeged_index(G):
    """Calculates Szeged index"""
    if not nx.is_connected(G):
        return False
    s = 0
    D = nx.floyd_warshall_numpy(G)
    for u, v in G.edges():
        diff = D[u, :] - D[v, :]
        s += (diff > 0).sum()*(diff < 0).sum()
    return float(s)


def harary_index(G):
    distances = nx.floyd_warshall_numpy(G)
    reciprocal_distances = np.reciprocal(distances)
    reciprocal_distances[np.isinf(reciprocal_distances)] = 0
    harary_index = 0.5 * np.sum(reciprocal_distances)
    return harary_index


def schultz_index(G):
    """ Schultz Index = Molecular Topological Index (mti) """
    A = nx.adjacency_matrix(G)
    D = np.diag(np.sum(A, axis=1))
    degree = np.array(list(G.degree()))[:, 1]
    mti = np.sum(degree * (A + D))
    return mti


def eccentric_connectivity_index(G):
    ec = nx.eccentricity(G)
    eci = sum(ec[v] * G.degree[v] for v in G)
    return eci


def augmented_valence_complexity(G):
    """ AVC from Randic and Plav """
    shortest_paths = dict(nx.shortest_path_length(G))
    avs = 0
    for u in G.nodes():
        valence = G.degree[u]
        for v, distance in shortest_paths[u].items():
            if distance != 0:
                avs += valence / distance
    return avs
