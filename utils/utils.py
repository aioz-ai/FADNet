import os
import json

import cvxpy as cp
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
from utils.metrics import RMSE

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from models.driving.FADNet import DrivingNet
from loaders.driving import get_iterator_driving
from loaders.complex_driving import get_iterator_complex_driving
import functools
import operator

EXTENSIONS = {"driving": ".npz"}

# Model size in bit
MODEL_SIZE_DICT = {"driving": 358116680}

# Model computation time in ms
COMPUTATION_TIME_DICT = {"driving": 30.2}

# Tags list
TAGS = ["Train/Loss", "Train/Acc", "Test/Loss", "Test/Acc", "Consensus"]


def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    args_string = ""

    args_to_show = ["experiment", "network_name", "fit_by_epoch", "bz_train",
                    "lr", "decay", "local_steps"]
    for arg in args_to_show:
        args_string += arg
        args_string += "_" + str(getattr(args, arg)) + "_"
    return args_string[:-1]


def get_optimal_mixing_matrix(adjacency_matrix, method="FDLA"):
    """

    :param adjacency_matrix: np.array()
    :param method:method to construct the mixing matrix weights;
     possible are:
      FMMC (Fast Mixin Markov Chain), see https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf
      FDLA (Fast Distributed Linear Averaging), https://web.stanford.edu/~boyd/papers/pdf/fastavg.pdf
    :return: optimal mixing matrix as np.array()
    """
    network_mask = 1 - adjacency_matrix
    N = adjacency_matrix.shape[0]

    s = cp.Variable()
    W = cp.Variable((N, N))
    objective = cp.Minimize(s)

    if method == "FDLA":
        constraints = [W == W.T,
                       W @ np.ones((N, 1)) == np.ones((N, 1)),
                       cp.multiply(W, network_mask) == np.zeros((N, N)),
                       -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
                       W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N)
                       ]
    elif method == "FMMC":
        constraints = [W == W.T,
                       W @ np.ones((N, 1)) == np.ones((N, 1)),
                       cp.multiply(W, network_mask) == np.zeros((N, N)),
                       -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
                       W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N),
                       np.zeros((N, N)) <= W
                       ]
    else:
        raise NotImplementedError

    prob = cp.Problem(objective, constraints)
    prob.solve()

    mixing_matrix = W.value

    mixing_matrix *= adjacency_matrix

    if method == "FMMC":
        mixing_matrix = np.multiply(mixing_matrix, mixing_matrix >= 0)

    # Force symmetry
    for i in range(N):
        if np.abs(np.sum(mixing_matrix[i, i:])) >= 1e-20:
            mixing_matrix[i, i:] *= (1 - np.sum(mixing_matrix[i, :(i)])) / np.sum(mixing_matrix[i, i:])
            mixing_matrix[i:, i] = mixing_matrix[i, i:]

    return mixing_matrix


def get_network(network_name, architecture, experiment):
    """
    load network  and generate mixing matrix,
    :param network_name: (str) should present in "graph_utils/data"
    :param architecture: possible are: "ring", "complete", "mst", "centralized" and "no_communication"
    :return: nx.DiGraph if architecture is "ring" and nx.DiGraph otherwise
    """
    if (architecture == "no_communication") or (architecture == "matcha") or (architecture == "matcha+"):
        path = os.path.join("graph_utils", "results", network_name, experiment, "original.gml")
    else:
        path = os.path.join("graph_utils", "results", network_name, experiment,  "{}.gml".format(architecture))

    if architecture == "ring":
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        mixing_matrix = mixing_matrix.astype(np.float64)
        mixing_matrix += np.eye(mixing_matrix.shape[0])
        mixing_matrix *= 0.5

        return nx.from_numpy_matrix(mixing_matrix, create_using=nx.DiGraph())

    elif architecture == "mct_plus":
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        n = network.number_of_nodes()
        mixing_matrix = mixing_matrix.astype(np.float64)
        mixing_matrix += np.eye(n)
        mixing_matrix = mixing_matrix / mixing_matrix.sum(axis=0)

        return nx.from_numpy_matrix(mixing_matrix, create_using=nx.DiGraph())

    elif architecture == "complete" or architecture == "matcha":
        network = nx.read_gml(path).to_undirected()

        n = network.number_of_nodes()
        mixing_matrix = np.ones((n, n)) / n

        return nx.from_numpy_matrix(mixing_matrix)

    elif architecture == "matcha+":
        network = nx.read_gml(path).to_undirected()

        return network

    elif architecture == "matcha+mst":
        path = os.path.join("graph_utils", "results", network_name, "mst.gml")
        return nx.read_gml(path).to_undirected()

    elif architecture == "matcha+ring":
        path = os.path.join("graph_utils", "results", network_name, "ring.gml")
        return nx.read_gml(path).to_undirected()

    elif architecture == "matcha+delta_mbst":
        path = os.path.join("graph_utils", "results", network_name, "mct_congest.gml")
        return nx.read_gml(path).to_undirected()

    elif architecture == "no_communication":
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        mixing_matrix = np.eye(network.number_of_nodes())

        return nx.from_numpy_matrix(mixing_matrix)
    else:
        network = nx.read_gml(path)
        mixing_matrix = nx.adjacency_matrix(network, weight=None).todense()

        adjacency_matrix = nx.adjacency_matrix(network, weight=None).todense()
        adjacency_matrix += np.eye(mixing_matrix.shape[0], dtype=np.int64)
        mixing_matrix = get_optimal_mixing_matrix(adjacency_matrix, method="FDLA")

        return nx.from_numpy_matrix(mixing_matrix)


def get_model(name, model, device, epoch_size, optimizer_name="adam", lr_scheduler="custom",
              initial_lr=1e-3, seed=1234):
    """
    Load Model object corresponding to the experiment
    :param name: experiment name; possible are: "driving" in name such as driving_carla, driving_gazebo
    :param device:
    :param epoch_size:
    :param optimizer_name: optimizer name, for now only "adam" is possible
    :param lr_scheduler:
    :param initial_lr:
    :param seed:
    :return: Model object
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if "driving" in name:
        criterion = nn.MSELoss()
        metric = [RMSE]
        return DrivingNet(model, criterion, metric, device, optimizer_name, lr_scheduler, initial_lr, epoch_size)
    else:
        raise NotImplementedError


def get_iterator(name, path, device, batch_size):
    if name == "driving_udacity":
        return get_iterator_driving(path, device, batch_size=batch_size)
    elif "driving_gazebo" in name or "driving_carla" in name:
        return get_iterator_complex_driving(path, device, batch_size=batch_size)
    else:
        raise NotImplementedError


def loggs_to_json(loggs_dir_path):
    """
    Write the results from logs folder as .json format
    :param loggs_dir_path: path to loggs folder
    """

    os.makedirs(os.path.join("results", "json"), exist_ok=True)
    all_results = {"Train/Loss": dict(), "Train/RMSE": dict(), "Test/Loss": dict(),
                   "Test/RMSE": dict(), "Consensus": dict(), "Round": dict(), "Train/Time": dict(), "Test/Time": dict()}

    for dname in os.listdir(loggs_dir_path):
        ea = EventAccumulator(os.path.join(loggs_dir_path, dname)).Reload()
        tags = ea.Tags()['scalars']

        for tag in tags:

            tag_values = []
            steps = []

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                steps.append(event.step)

            all_results[tag][dname] = tag_values
            all_results["Round"][dname] = steps

    json_path = os.path.join("results", "json", "{}.json".format(os.path.split(loggs_dir_path)[1]))
    with open(json_path, "w") as f:
        json.dump(all_results, f)

class logger_write_params(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)

def print_model(model, logger):
    print('>>>>>>>>>> Network Architecture')
    print(model)
    nParams = 0
    for w in model.parameters():
        nParams += functools.reduce(operator.mul, w.size(), 1)
    if logger:
        logger.write('nParams=\t'+str(nParams))
