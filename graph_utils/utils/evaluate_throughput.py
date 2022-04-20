import networkx as nx
import numpy as np

np.seterr(all="ignore")


def cycle_time_decision(G, lambda_0):
    """
    Answers the cycle time decision problem question: Is the throughput of G at most lambda ?
    :param G: (nx.DiGraph) Strong Weighted Digraph
    :param lambda_0: (numerical)
    """
    A = nx.adjacency_matrix(G).toarray()
    new_A = lambda_0 - A

    new_G = nx.from_numpy_matrix(new_A, create_using=nx.DiGraph())

    answer = True
    try:
        nx.bellman_ford_predecessor_and_distance(new_G, 0)
    except nx.NetworkXUnbounded:
        answer = False
    return answer


def evaluate_cycle_time(G, s=0):
    """
    Evaluate the cycle time of a strong weighted digraph. For now the implementation only supports integer delays
    :param G: (nx.DiGraph) strong weighted digraph
    :param s: starting point
    :return: lambda_G
            The cycle time of G
    """
    n = len(G)
    nodes_to_indices = {node: idx for idx, node in enumerate(G.nodes)}

    # Head
    D = np.zeros((n + 1, n)) - np.inf
    pi = np.zeros((n + 1, n), dtype=np.int64) - 1
    D[0, s] = 0

    # Body
    for k in range(1, n + 1):
        for v in G.nodes:
            for u in G.predecessors(v):
                if D[k, nodes_to_indices[v]] < D[k - 1, nodes_to_indices[u]] + G.get_edge_data(u, v)['weight']:
                    D[k, nodes_to_indices[v]] = D[k - 1, nodes_to_indices[u]] \
                                                + G.get_edge_data(u, v)['weight']

                    pi[k, nodes_to_indices[v]] = nodes_to_indices[u]

    # Tail
    lambda_ = -np.inf
    M = np.zeros((n,)) + np.inf
    K = np.zeros((n,), dtype=np.int64) - 1
    for v in G.nodes:
        for k in range(0, n):
            if M[nodes_to_indices[v]] > (D[n, nodes_to_indices[v]] - D[k, nodes_to_indices[v]]) / (n - k):
                M[nodes_to_indices[v]] = (D[n, nodes_to_indices[v]] - D[k, nodes_to_indices[v]]) / (n - k)
                K[nodes_to_indices[v]] = k

        if lambda_ < M[nodes_to_indices[v]]:
            lambda_ = M[nodes_to_indices[v]]
            v_star = nodes_to_indices[v]

    # Get critical cycle
    path = []
    actual = v_star
    for i in range(n, -1, -1):
        path.append(actual)
        actual = pi[i, actual]

    path.reverse()

    return lambda_, path, n - K[v_star]


def evaluate_throughput(G):
    """
    Evaluate the throughput of a strong weighted digraph. For now the implementation only supports integer delays
    :param G: (nx.DiGraph) strong weighted digraph
    :return: The throughput of G
    """
    lambda_, _, _ = evaluate_cycle_time(G)
    return 1 / lambda_
