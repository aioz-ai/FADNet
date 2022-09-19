import os
import argparse

import networkx as nx

from utils.evaluate_throughput import evaluate_cycle_time
from utils.utils import get_connectivity_graph, add_upload_download_delays, get_delta_mbst_overlay,\
    get_star_overlay, get_ring_overlay, get_matcha_cycle_time

# Model size in bit
MODEL_SIZE_DICT = {"synthetic": 4354,
                   "shakespeare": 3385747,
                   "femnist": 4843243,
                   "sent140": 19269416,
                   "inaturalist": 44961717,
                   "driving_gazebo": 10091936,
                   "driving_carla": 10091936,
                   "driving_udacity": 10091936}

# Model computation time in ms
COMPUTATION_TIME_DICT = {"synthetic": 1.5,
                         "shakespeare": 389.6,
                         "femnist": 4.6,
                         "sent140": 9.8,
                         "inaturalist": 25.4,
                         "driving_gazebo": 4.9,
                         "driving_carla": 7.2,
                         "driving_udacity": 0.66}


parser = argparse.ArgumentParser()

parser.add_argument('name',
                    help='name of the network to use;')
parser.add_argument("--experiment",
                    type=str,
                    help="name of the experiment that will be run on the network;"
                         "possible are femnist, inaturalist, synthetic, shakespeare, sent140, driving_gazebo, driving_carla;"
                         "if not precised --model_size will be used as model size;",
                    default=None)
parser.add_argument('--model_size',
                    type=float,
                    help="size of the model that will be transmitted on the network in bit;"
                         "ignored if --experiment is precised;",
                    default=1e8)
parser.add_argument("--local_steps",
                    type=int,
                    help="number of local steps, used to get computation time",
                    default=1)
parser.add_argument("--upload_capacity",
                    type=float,
                    help="upload capacity at edge in bit/s; default=1e32",
                    default=1e32)
parser.add_argument("--download_capacity",
                    type=float,
                    help="download capacity at edge in bit/s; default=1e32",
                    default=1e32)
parser.add_argument("--communication_budget",
                    type=float,
                    help="communication budget to use with matcha; will be ignored if name is not matcha",
                    default=0.5)
parser.add_argument("--default_capacity",
                    type=float,
                    help="default capacity (in bit/s) to use on links with unknown capacity",
                    default=1e9)
parser.add_argument('--centrality',
                    help="centrality type; default: load;",
                    default="load")

parser.set_defaults(user=False)

args = parser.parse_args()
args.default_capacity *= 1e-3

if __name__ == "__main__":
    if args.experiment is not None:
        args.model_size = MODEL_SIZE_DICT[args.experiment]
        args.computation_time = args.local_steps * COMPUTATION_TIME_DICT[args.experiment]

    upload_delay = (args.model_size / args.upload_capacity) * 1e3
    download_delay = (args.model_size / args.download_capacity) * 1e3

    result_dir = "./results/{}/{}".format(args.name, args.experiment)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    results_txt_path = os.path.join(result_dir, "cycle_time.txt")
    results_file = open(results_txt_path, "w")

    path_to_graph = "./data/{}.gml".format(args.name)

    underlay = nx.read_gml(path_to_graph)

    print("Number of Workers: {}".format(underlay.number_of_nodes()))
    print("Number of links: {}".format(underlay.number_of_edges()))

    nx.set_node_attributes(underlay, upload_delay, 'uploadDelay')
    nx.set_node_attributes(underlay, download_delay, "downloadDelay")

    nx.write_gml(underlay.copy(), os.path.join(result_dir, "original.gml"))

    connectivity_graph = get_connectivity_graph(underlay, args.default_capacity)

    # MST
    for u, v, data in connectivity_graph.edges(data=True):
        weight = args.computation_time + data["latency"] + args.model_size / data["availableBandwidth"]
        connectivity_graph.add_edge(u, v, weight=weight)

    MST = nx.minimum_spanning_tree(connectivity_graph.copy(), weight="weight")

    MST = MST.to_directed()

    cycle_time, _, _ = evaluate_cycle_time(add_upload_download_delays(MST, args.computation_time, args.model_size))

    nx.write_gml(MST, os.path.join(result_dir, "mst.gml"))
    print("Cycle time for MST architecture: {0:.1f}".format(cycle_time))
    results_file.write("MST {}\n".format(cycle_time))

    # delta-MBST
    delta_mbst, best_cycle_time, best_delta = \
        get_delta_mbst_overlay(connectivity_graph.copy(), args.computation_time, args.model_size)

    delta_mbst = add_upload_download_delays(delta_mbst, args.computation_time, args.model_size)
    cycle_time, _, _ = evaluate_cycle_time(delta_mbst)

    nx.write_gml(delta_mbst, os.path.join(result_dir, "mct_congest.gml"))
    print("Cycle time for delta-MBST architecture: {0:.1f} ms".format(cycle_time))
    results_file.write("MCT_congest {}\n".format(cycle_time))

    # Star
    star = get_star_overlay(connectivity_graph.copy(), args.centrality)

    cycle_time, _, _ = evaluate_cycle_time(add_upload_download_delays(star, args.computation_time, args.model_size))

    cycle_time = (cycle_time - args.computation_time) * 2 + args.computation_time

    nx.write_gml(star, os.path.join(result_dir, "centralized.gml"))
    print("Cycle time for STAR architecture: {0:.1f} ms".format(cycle_time))
    results_file.write("Server {}\n".format(cycle_time))

    # Ring
    ring = get_ring_overlay(connectivity_graph.copy(), args.computation_time, args.model_size)

    cycle_time, _, _ = evaluate_cycle_time(add_upload_download_delays(ring, args.computation_time, args.model_size))

    nx.write_gml(ring, os.path.join(result_dir, "ring.gml"))
    print("Cycle time for RING architecture: {0:.1f} ms".format(cycle_time))
    results_file.write("Ring graph {}\n".format(cycle_time))

    # MATCHA
    cycle_time = get_matcha_cycle_time(underlay.copy(), connectivity_graph.copy(),
                                       args.computation_time, args.model_size, args.communication_budget)

    print("Cycle time for MATCHA architecture: {0:.1f} ms".format(cycle_time))
    results_file.write("MATCHA {}\n".format(cycle_time))

    # MATCHA+
    cycle_time = get_matcha_cycle_time(connectivity_graph.copy(), connectivity_graph.copy(),
                                       args.computation_time, args.model_size, args.communication_budget)

    print("Cycle time for MATCHA+ architecture: {0:.1f} ms".format(cycle_time))
    results_file.write("MATCHA {}\n".format(cycle_time))
