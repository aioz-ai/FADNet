import os
import csv
import shutil
import random

import networkx as nx
import numpy as np

from .evaluate_throughput import evaluate_cycle_time
from .mbst import cube_algorithm, delta_prim
from .tsp_christofides import christofides_tsp
from .matcha import RandomTopologyGenerator
from .matching_decomposition import get_matching_list_from_graph


def get_connectivity_graph(underlay, default_capacity=1e9):
    """

    :param underlay:
    :param default_capacity:
    :return:
    """
    connectivity_graph = nx.Graph()
    connectivity_graph.add_nodes_from(underlay.nodes(data=True))

    dijkstra_result = nx.all_pairs_dijkstra(underlay.copy(), weight="distance")

    for node, (weights_dict, paths_dict) in dijkstra_result:
        for neighbour in paths_dict.keys():
            if node != neighbour:
                path = paths_dict[neighbour]

                distance = 0.
                for idx in range(len(path) - 1):
                    u = path[idx]
                    v = path[idx + 1]

                    data = underlay.get_edge_data(u, v)
                    distance += data["distance"]

                available_bandwidth = default_capacity / (len(path) - 1)

                latency = 0.0085 * distance + 4

                connectivity_graph.add_edge(node, neighbour, availableBandwidth=available_bandwidth, latency=latency)

    return connectivity_graph


def add_upload_download_delays(overlay, computation_time, model_size):
    """
    Takes as input an nx.Graph(), each edge should have attributes "latency" and "availableBandwidth";
    each node should have attribute "uploadDelay" and "downloadDelay";
    The weight (delay) of edge (i, j) is computed as:
    d(i, j) = computation_time + latency(i, j) + max(M/[availableBandwidth(i, j), "uploadDelay", "downloadDelay"]$$
    :param overlay:
    :param computation_time:
    :param model_size:
    :return:
    """
    overlay = overlay.to_directed()

    out_degree_dict = dict(overlay.out_degree)
    in_degree_dict = dict(overlay.in_degree)

    for u, v, data in overlay.edges(data=True):
        upload_delay = out_degree_dict[u] * overlay.nodes[u]["uploadDelay"]
        download_delay = in_degree_dict[v] * overlay.nodes[v]["downloadDelay"]

        weight = computation_time + data["latency"] +\
            max(upload_delay, download_delay, model_size/data["availableBandwidth"])

        overlay.add_edge(u, v, weight=weight)

    return overlay


def get_star_overlay(connectivity_graph, centrality):
    """
    Generate server connectivity graph given an underlay topology represented as an nx.Graph
    :param connectivity_graph: nx.Graph() object, each edge should have availableBandwidth:
     "latency", "availableBandwidth" and "weight";
    :param centrality: mode of centrality to use, possible: "load", "distance", "information", default="load"
    :return: nx.Graph()
    """
    if centrality == "distance":
        centrality_dict = nx.algorithms.centrality.closeness_centrality(connectivity_graph, distance="latency")
        server_node = max(centrality_dict, key=centrality_dict.get)

    elif centrality == "information":
        centrality_dict = nx.algorithms.centrality.information_centrality(connectivity_graph, weight="latency")
        server_node = max(centrality_dict, key=centrality_dict.get)

    else:
        # centrality = load_centrality
        centrality_dict = nx.algorithms.centrality.load_centrality(connectivity_graph, weight="latency")
        server_node = max(centrality_dict, key=centrality_dict.get)

    weights, paths = nx.single_source_dijkstra(connectivity_graph, source=server_node, weight="weight")

    star = nx.Graph()
    star.add_nodes_from(connectivity_graph.nodes(data=True))

    for node in paths.keys():
        if node != server_node:

            latency = 0.
            available_bandwidth = 1e32
            for idx in range(len(paths[node]) - 1):
                u = paths[node][idx]
                v = paths[node][idx + 1]

                data = connectivity_graph.get_edge_data(u, v)
                latency += data["latency"]
                available_bandwidth = data["availableBandwidth"]

            star.add_edge(server_node, node, availableBandwidth=available_bandwidth, latency=latency)

    return star


def get_ring_overlay(connectivity_graph, computation_time, model_size):
    """

    :param connectivity_graph:
    :param computation_time:
    :param model_size:
    :return:
    """
    for u, v, data in connectivity_graph.edges(data=True):
        upload_delay = connectivity_graph.nodes[u]["uploadDelay"]
        download_delay = connectivity_graph.nodes[v]["downloadDelay"]

        weight = computation_time + data["latency"] + max(upload_delay,
                                                          download_delay,
                                                          model_size / data["availableBandwidth"])

        connectivity_graph.add_edge(u, v, weight=weight)

    adjacency_matrix = nx.adjacency_matrix(connectivity_graph, weight="weight").toarray()
    tsp_nodes = christofides_tsp(adjacency_matrix)

    ring = nx.DiGraph()
    ring.add_nodes_from(connectivity_graph.nodes(data=True))

    for idx in range(len(tsp_nodes) - 1):
        # get the label of source and sink nodes from the original graph
        source_node = list(connectivity_graph.nodes())[tsp_nodes[idx]]
        sink_node = list(connectivity_graph.nodes())[tsp_nodes[idx + 1]]

        ring.add_edge(source_node, sink_node,
                      latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                      availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)['availableBandwidth'],
                      weight=connectivity_graph.get_edge_data(source_node, sink_node)['weight'])

    # add final link to close the circuit
    source_node = list(connectivity_graph.nodes())[tsp_nodes[-1]]
    sink_node = list(connectivity_graph.nodes())[tsp_nodes[0]]
    ring.add_edge(source_node, sink_node,
                  latency=connectivity_graph.get_edge_data(source_node, sink_node)['latency'],
                  availableBandwidth=connectivity_graph.get_edge_data(source_node, sink_node)['availableBandwidth'],
                  weight=connectivity_graph.get_edge_data(source_node, sink_node)['weight'])

    return ring


def generate_random_ring(list_of_nodes):
    """
    Generate a random ring graph connecting a list of nodes
    :param list_of_nodes:
    :return: nx.DiGraph()
    """
    ring = nx.DiGraph()

    ring.add_nodes_from(list_of_nodes)

    random.shuffle(list_of_nodes)

    for idx in range(len(list_of_nodes) - 1):
        # get the label of source and sink nodes from the original graph
        source_node = list_of_nodes[idx]
        sink_node = list_of_nodes[idx + 1]

        ring.add_edge(source_node, sink_node)

    # add final link to close the circuit
    source_node = list_of_nodes[-1]
    sink_node = list_of_nodes[0]
    ring.add_edge(source_node, sink_node)

    mixing_matrix = nx.adjacency_matrix(ring, weight=None).todense().astype(np.float64)

    mixing_matrix += np.eye(mixing_matrix.shape[0])
    mixing_matrix *= 0.5

    return nx.from_numpy_matrix(mixing_matrix, create_using=nx.DiGraph())


def get_delta_mbst_overlay(connectivity_graph, computation_time, model_size):
    """

    :param connectivity_graph:
    :param computation_time:
    :param model_size:
    :return:
    """
    for u, v, data in connectivity_graph.edges(data=True):
        weight = computation_time + data["latency"] + \
                 max(connectivity_graph.nodes[u]["uploadDelay"], connectivity_graph.nodes[v]["downloadDelay"],
                     model_size / data["availableBandwidth"]) + \
                 max(connectivity_graph.nodes[v]["uploadDelay"], connectivity_graph.nodes[u]["downloadDelay"],
                     model_size / data["availableBandwidth"])

        connectivity_graph.add_edge(u, v, weight=weight, latency=data["latency"],
                                    availableBandwidth=data["availableBandwidth"])

    for u in connectivity_graph.nodes:
        connectivity_graph.add_edge(u, u, weight=0, latency=0, availableBandwidth=1e32)

    best_result = cube_algorithm(connectivity_graph.copy()).to_directed()

    for u, v in best_result.edges:
        best_result.add_edge(u, v,
                             latency=connectivity_graph.get_edge_data(u, v)['latency'],
                             availableBandwidth=connectivity_graph.get_edge_data(u, v)['availableBandwidth'])

    best_cycle_time, _, _ = evaluate_cycle_time(add_upload_download_delays(best_result, computation_time, model_size))
    best_delta = 2

    n_nodes = connectivity_graph.number_of_nodes()
    for delta in range(2, n_nodes):
        result = delta_prim(connectivity_graph.copy(), delta).to_directed()

        for u, v, data in result.edges(data=True):
            weight = data["weight"] - (result.nodes[u]["uploadDelay"] + result.nodes[v]["downloadDelay"])

            result.add_edge(u, v, weight=weight,
                            latency=connectivity_graph.get_edge_data(u, v)['latency'],
                            availableBandwidth=connectivity_graph.get_edge_data(u, v)['availableBandwidth'])

        cycle_time, _, _ = evaluate_cycle_time(add_upload_download_delays(result, computation_time, model_size))

        if cycle_time < best_cycle_time:
            best_result = result
            best_cycle_time = cycle_time
            best_delta = delta

    return best_result, best_cycle_time, best_delta


def get_matcha_cycle_time(underlay, connectivity_graph, computation_time, model_size, communication_budget):
    """

    :param underlay:
    :param connectivity_graph:
    :param computation_time:
    :param model_size:
    :param communication_budget:
    :return:
    """
    path_to_save_network = os.path.join("temp", "colored_network.gml")
    path_to_matching_history_file = os.path.join("temp", "matching_history.csv")

    try:
        shutil.rmtree("temp")
    except FileNotFoundError:
        pass

    os.makedirs("temp", exist_ok=True)

    topology_generator = RandomTopologyGenerator(underlay.copy(),
                                                 communication_budget,
                                                 network_save_path=path_to_save_network,
                                                 path_to_history_file=path_to_matching_history_file)

    n_rounds = 1000
    np.random.seed(0)
    for _ in range(n_rounds):
        topology_generator.step()

    path_to_colored_network = os.path.join("temp", "colored_network.gml")
    path_to_matching_history_file = os.path.join("temp", "matching_history.csv")

    colored_network = nx.read_gml(path_to_colored_network)
    matching_list = get_matching_list_from_graph(colored_network)

    simulated_time = np.zeros(n_rounds)
    with open(path_to_matching_history_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')

        for ii, row in enumerate(csv_reader):
            overlay = nx.Graph()
            overlay.add_nodes_from(connectivity_graph.nodes(data=True))

            current_matching_activations = row
            for matching_idx, matching_activation in enumerate(current_matching_activations):
                if int(matching_activation):
                    overlay = nx.compose(overlay, matching_list[matching_idx])

            for u, v in overlay.edges():
                overlay.add_edge(u, v,
                                 latency=connectivity_graph.get_edge_data(u, v)["latency"],
                                 availableBandwidth=connectivity_graph.get_edge_data(u, v)['availableBandwidth']
                                 )

            if nx.is_empty(overlay):
                # If overlay is empty, then no communication cost is added
                simulated_time[:, ii] = computation_time

            else:
                overlay = add_upload_download_delays(overlay, computation_time, model_size)

                cycle_time = 0
                for u, v, data in overlay.edges(data=True):
                    if data["weight"] > cycle_time:
                        cycle_time = data["weight"]

                simulated_time[ii] = cycle_time

    simulated_time = simulated_time.cumsum()

    try:
        shutil.rmtree("temp")
    except FileNotFoundError:
        pass

    return simulated_time[-1] / (n_rounds - 1)
