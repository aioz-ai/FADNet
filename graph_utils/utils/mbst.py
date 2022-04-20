import networkx as nx
import numpy as np
from networkx.algorithms.tournament import hamiltonian_path


def cube_algorithm(G_complete):
    """
    Use cube algorithm to build an approximation for the 2-MBST problem on G:
    1. Add edges to G to build complete graph G_complete
    2. Build an MST T of G_complete
    3. Build the the cube of T
    4. find a Hamiltonian path in the cube of T
    :param G : (nx.Graph())
    """
    T = nx.minimum_spanning_tree(G_complete, weight="weight")

    T_cube = nx.Graph()
    T_cube.add_nodes_from(T.nodes(data=True))

    shortest_paths = nx.shortest_path_length(T)
    for source, lengths_dict in shortest_paths:
        for target in lengths_dict:
            if lengths_dict[target] <= 3:
                T_cube.add_edge(source, target,
                                weight=G_complete.get_edge_data(source, target)["weight"])

    ham_path = hamiltonian_path(T_cube.to_directed())

    result = nx.Graph()
    result.add_nodes_from(G_complete.nodes(data=True))

    for idx in range(len(ham_path) - 1):
        result.add_edge(ham_path[idx], ham_path[idx + 1],
                        weight=G_complete.get_edge_data(ham_path[idx], ham_path[idx + 1])['weight'])

    return result


def delta_prim(G_complete, delta):
    """
    implementation of delta prim algorithm from https://ieeexplore.ieee.org/document/850653
    :param G: (nx.Graph())
    :param delta: (int)
    :return: a tree T with degree at most delta
    """
    N = G_complete.number_of_nodes()
    T = nx.Graph()

    T.add_node(list(G_complete.nodes)[0])

    while len(T.edges) < N - 1:
        smallest_weight = np.inf
        edge_to_add = None
        for u in T.nodes:
            for v in G_complete.nodes:
                if (v not in T.nodes) and (T.degree[u] < delta):
                    weight = G_complete.get_edge_data(u, v)["weight"]
                    if weight < smallest_weight:
                        smallest_weight = weight
                        edge_to_add = (u, v)

        T.add_edge(*edge_to_add, weight=smallest_weight)

    T.add_nodes_from(G_complete.nodes(data=True))

    return T
