import itertools
from random import randint

import numpy as np
import networkx as nx

from networkx.algorithms.matching import max_weight_matching
from networkx.algorithms.euler import eulerian_circuit


def christofides_tsp(graph, starting_node=0):
    """
    Christofides TSP algorithm
    http://www.dtic.mil/dtic/tr/fulltext/u2/a025602.pdf
    Args:
        graph: 2d numpy array matrix
        starting_node: of the TSP
    Returns:
        tour given by christofies TSP algorithm
    Examples:
        >>> import numpy as np
        >>> graph = np.array([[  0, 300, 250, 190, 230],
        >>>                   [300,   0, 230, 330, 150],
        >>>                   [250, 230,   0, 240, 120],
        >>>                   [190, 330, 240,   0, 220],
        >>>                   [230, 150, 120, 220,   0]])
        >>> christofides_tsp(graph)
    """

    mst = minimal_spanning_tree(graph, 'Prim', starting_node=0)
    odd_degree_nodes = list(_get_odd_degree_vertices(mst))
    odd_degree_nodes_ix = np.ix_(odd_degree_nodes, odd_degree_nodes)
    nx_graph = nx.from_numpy_array(-1 * graph[odd_degree_nodes_ix])
    matching = max_weight_matching(nx_graph, maxcardinality=True)
    euler_multigraph = nx.MultiGraph(mst)
    for edge in matching:
        euler_multigraph.add_edge(odd_degree_nodes[edge[0]], odd_degree_nodes[edge[1]],
                                  weight=graph[odd_degree_nodes[edge[0]]][odd_degree_nodes[edge[1]]])
    euler_tour = list(eulerian_circuit(euler_multigraph, source=starting_node))
    path = list(itertools.chain.from_iterable(euler_tour))
    return _remove_repeated_vertices(path, starting_node)[:-1]


def _get_odd_degree_vertices(graph):
    """
    Finds all the odd degree vertices in graph
    Args:
        graph: 2d np array as adj. matrix
    Returns:
    Set of vertices that have odd degree
    """
    odd_degree_vertices = set()
    for index, row in enumerate(graph):
        if len(np.nonzero(row)[0]) % 2 != 0:
            odd_degree_vertices.add(index)
    return odd_degree_vertices


def _remove_repeated_vertices(path, starting_node):
    path = list(dict.fromkeys(path).keys())
    path.append(starting_node)
    return path


def minimal_spanning_tree(graph, mode='Prim', starting_node=None):
    """
    Args:
        graph:  weighted adjacency matrix as 2d np.array
        mode: method for calculating minimal spanning tree
        starting_node: node number to start construction of minimal spanning tree (Prim)
    Returns:
        minimal spanning tree as 2d array
    """

    if mode == 'Prim':
        return _minimal_spanning_tree_prim(graph, starting_node)


def _minimal_spanning_tree_prim(graph, starting_node):
    """
    Args:
        graph: weighted adj. matrix as 2d np.array
        starting_node: node number to start construction of minimal spanning tree
    Returns:
        minimal spanning tree as 2d array calculted by Prim
    """

    node_count = len(graph)
    all_nodes = [i for i in range(node_count)]

    if starting_node is None:
        starting_node = randint(0, node_count-1)

    unvisited_nodes = all_nodes
    visited_nodes = [starting_node]
    unvisited_nodes.remove(starting_node)
    mst = np.zeros((node_count, node_count))

    while len(visited_nodes) != node_count:
        selected_subgraph = graph[np.array(visited_nodes)[:, None], np.array(unvisited_nodes)]
        # we mask non-exist edges with -- so it doesn't crash the argmin
        min_edge_index = np.unravel_index(np.ma.masked_equal(selected_subgraph, 0, copy=False).argmin(),
                                          selected_subgraph.shape)
        edge_from = visited_nodes[min_edge_index[0]]
        edge_to = unvisited_nodes[min_edge_index[1]]
        mst[edge_from, edge_to] = graph[edge_from, edge_to]
        mst[edge_to, edge_from] = graph[edge_from, edge_to]
        unvisited_nodes.remove(edge_to)
        visited_nodes.append(edge_to)
    return mst