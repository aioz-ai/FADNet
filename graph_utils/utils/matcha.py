import os
import csv

import cvxpy as cp
import numpy as np
import networkx as nx

from .matching_decomposition import matching_decomposition


class RandomTopologyGenerator(object):
    """
    Attributes:
        - laplacian_matrices: List of numpy arrays; each array represents the laplacian matrix of a matching;
        - communication_budget: Constraint controlling the sum of the weights,
         and equivalently controlling the expected communication time;
        - path_to_history_file: path to .csv file used to save the history of selected matching at each step
        - activation_probas: np.array of the same size as "laplacian_matrices";
        - current_matching_activations: list of  booleans, each of them represent if a matching is used;
        - matching_list: list of nx.Graph() objects;
        - alpha: float to be use in generating mixing matrix
    """
    def __init__(self, network, communication_budget, network_save_path=None, path_to_history_file=None):
        self.network = network
        self.communication_budget = communication_budget
        self.path_to_history_file = path_to_history_file
        self.network_save_path = network_save_path

        # eliminate self loops
        self.network.remove_edges_from(nx.selfloop_edges(self.network))

        self.matching_list, self.laplacian_matrices = matching_decomposition(self.network)

        self.number_workers = self.laplacian_matrices[0].shape[0]
        self.number_matching = len(self.laplacian_matrices)

        # Initialize generator parameters
        self.activation_probas = self.get_matching_activation_probabilities()
        self.activation_probas = np.clip(self.activation_probas, 0., 1.)

        self.alpha, self.spectral_norm = self.get_mixing_matrix_parameter()

        # Initialize
        self.current_step = -1
        self.current_matching_activations = np.ones(self.number_workers)
        self.current_topology = self.network

        if self.network_save_path:
            nx.write_gml(self.network, self.network_save_path)

    def get_matching_activation_probabilities(self):
        """
        Computes a set of activation probabilities that maximize the connectivity of the expected graph
         given a communication time constraint;
        For given Laplacian matrices, it computes optimal weights to sum them, in order to maximize
         the second largest eigenvalue of their weighted sum;
        See https://arxiv.org/pdf/1905.09435.pdf  (Formula 5) for details;
         and equivalently controlling the expected communication time;
        :return: np.array of the same size as "laplacian_matrices"; each entry represents the probability
         of activating a sub-graph;
        """
        p = cp.Variable(self.number_matching)
        gamma = cp.Variable()
        beta = cp.Variable()
        constraints = [p <= 1, p >= 0,
                       p.T @ np.ones(self.number_matching) <= self.communication_budget * self.number_matching,
                       gamma * np.eye(self.number_workers) - beta * np.ones((self.number_workers, self.number_workers))
                       << cp.sum([p[i] * self.laplacian_matrices[i] for i in range(self.number_matching)])]
        objective = cp.Maximize(gamma)
        problem = cp.Problem(objective, constraints)

        problem.solve()

        return p.value

    def get_mixing_matrix_parameter(self):
        """
        Computes optimal equal weight mixing matrix parameter;
         i.e. computes alpha in order to optimize the spectral gap of the mixing matrix W, where
         W = I - alpha * L_bar, with being identity matrix and L_bar is the expected Laplacian matrix;
         See https://arxiv.org/pdf/1905.09435.pdf  (Formula 6 and 7) for details;
         each entry represents the probability of activating a sub-graph;
        :return: alpha (float)
        """
        L_bar = np.zeros((self.number_workers, self.number_workers))
        L_tilde = np.zeros((self.number_workers, self.number_workers))

        for idx in range(self.number_matching):
            L_bar += self.activation_probas[idx] * self.laplacian_matrices[idx]
            L_tilde += self.activation_probas[idx] * (1 - self.activation_probas[idx]) * self.laplacian_matrices[idx]

        rho = cp.Variable()
        alpha = cp.Variable()
        beta = cp.Variable()

        objective = cp.Minimize(rho)

        constraints = [alpha ** 2 - beta <= 0,
                       np.eye(self.number_workers) - 2 * alpha * L_bar + beta * (L_bar @ L_bar + 2 * L_tilde)
                       - (1 / self.number_workers) * np.ones((self.number_workers, self.number_workers))
                       << rho * np.eye(self.number_workers)]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return alpha.value, rho.value

    def step(self):
        """
         Generating random topology at any iteration: given activation probabilities, generates an independent
          Bernoulli random variable Bj for each matching  in "matching_list",
           the activated topology is the concatenation of the activated matching.
            The mixing matrix is then computed as W = I - alpha * L, where L is the Laplacian matrix
            of the activated topology;
         """
        self.current_topology = nx.Graph()
        laplacian_matrix = np.zeros((self.number_workers, self.number_workers))

        self.current_matching_activations = np.random.binomial(n=1, p=self.activation_probas)
        while self.current_matching_activations.sum() == 0:
            self.current_matching_activations = np.random.binomial(n=1, p=self.activation_probas)

        for idx, matching_activation in enumerate(self.current_matching_activations):
            if matching_activation:
                self.current_topology = nx.compose(self.current_topology, self.matching_list[idx])
                laplacian_matrix += self.laplacian_matrices[idx]

        mixing_matrix = np.eye(self.number_workers) - self.alpha * laplacian_matrix
                
        self.current_topology = nx.from_numpy_matrix(mixing_matrix)

        self.current_step += 1

        if self.path_to_history_file:
            with open(self.path_to_history_file, "a") as csvfile:
                writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(self.current_matching_activations.tolist())
