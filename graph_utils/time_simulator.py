import time

import networkx as nx
import numpy as np

import geopy.distance
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="delay", timeout=20)


def get_zoo_topology(file_path,
                     bandwidth=1e9,
                     upload_capacity_at_edge=35 * 1e6,
                     download_capacity_at_edge=144 * 1e6):
    """
    Read zoo_topology data into nx.DiGraph();
     in the output graph each edge has two information: "capacity" and "distance";
     each node has two information: "upload capacity" and "download capacity";
    :param file_path : path to .gml file with topology information
    :param bandwidth: (float) represent links capacity,
                     used when information not available in .gml file
    :param upload_capacity_at_edge: https://en.wikipedia.org/wiki/Bit_rate for information
    :param download_capacity_at_edge: https://en.wikipedia.org/wiki/Bit_rate for information
    :return:  G_z (nx.DiGraph)
    """

    network_data = nx.read_gml(file_path)

    G_z = nx.Graph()
    G_z.add_nodes_from(network_data)

    # add nodes capacity
    nx.set_node_attributes(G_z, upload_capacity_at_edge * 1e-3, 'upload_capacity')
    nx.set_node_attributes(G_z, download_capacity_at_edge * 1e-3, "download_capacity")

    # add edges data
    for u, v, data in network_data.edges.data():
        # get distance
        try:
            distance = data["distance"]

        except AttributeError:
            try:
                coords_1 = (network_data.nodes(data=True)[u]["Latitude"],
                            network_data.nodes(data=True)[u]["Longitude"])

                coords_2 = (network_data.nodes(data=True)[v]["Latitude"],
                            network_data.nodes(data=True)[v]["Longitude"])

            except KeyError:
                time.sleep(1.2)  # To avoid Service time out Error

                geo = geolocator.geocode(u, timeout=20)

                coords_1 = (geo.latitude, geo.longitude)

                time.sleep(1.2)  # To avoid Service time out Error

                geo = geolocator.geocode(v, timeout=20)

                coords_2 = (geo.latitude, geo.longitude)

            distance = geopy.distance.distance(coords_1, coords_2).km

        # add_edge
        G_z.add_edge(u, v, capacity=bandwidth * 1e-3, distance=distance)

    return G_z


def initialize_delays(underlay, overlay, model_size):
    """
    compute delays between nodes ignoring download congestion effect
    :param underlay: (nx.Graph())
    :param overlay: (nx.Graph())
    :param model_size: message_size in bits, see https://keras.io/applications/ for examples
    :return: nxGraph()
    """
    for u, v, data in overlay.edges(data=True):
        overlay.edges[u, v]["delay"] = overlay.edges[u, v]["weight"]

    return overlay


def init_iteration_end_time(overlay, computation_time=0):
    """

    :param overlay:
    :param computation_time:
    :return:
    """
    nx.set_node_attributes(overlay, computation_time, "end_time")
    return overlay


def get_iteration_end_time(underlay, overlay, model_size, computation_time):
    """
    Compute the end times of next iteration having the  end times for current iteration.
    :param underlay:
    :param overlay:
    :param model_size:
    :param computation_time
    :return:
    """
    out_degrees = dict(overlay.out_degree())
    for i, j in overlay.edges:
        overlay.edges[i, j]["t"] = overlay.edges[i, j]["delay"] + overlay.nodes[i]["end_time"]

    def get_edge_time(e):
        return overlay.edges[e[0], e[1]]["t"]

    for j in overlay.nodes:
        overlay.nodes[j]["end_time"] = 0

        # get all the input edges to "j" sorted by t_{ij}
        edges = []
        for i in overlay.predecessors(j):
            edges.append((i, j))

        if len(edges) > 0:
            edges.sort(key=get_edge_time)

            t_prev = get_edge_time(edges[0]) + model_size / underlay.nodes[j]["download_capacity"]

            for edge in edges[1:]:
                if get_edge_time(edge) <= t_prev + model_size / underlay.nodes[j]["download_capacity"]:
                    t_prev = t_prev + model_size / underlay.nodes[j]["download_capacity"]
                else:
                    t_prev = get_edge_time(edge)

        else:
            t_prev = 0

        overlay.nodes[j]["end_time"] = t_prev + computation_time + \
                                       (model_size * out_degrees[j]) / underlay.nodes[j]["upload_capacity"]

    return overlay


def simulate_network(underlay, overlay, n_iterations, model_size=1e8, computation_time=0):
    """

    :param underlay:
    :param overlay:
    :param n_iterations:
    :param model_size:
    :param computation_time
    :return:
    """
    time_evolution = np.zeros((overlay.number_of_nodes(), n_iterations))

    overlay = initialize_delays(underlay, overlay, model_size)
    overlay = init_iteration_end_time(overlay, computation_time)

    for iteration in range(n_iterations):
        overlay = get_iteration_end_time(underlay, overlay, model_size, computation_time)
        for ii, (_, end_time) in enumerate(overlay.nodes.data("end_time")):
            time_evolution[ii, iteration] = end_time

    return time_evolution
