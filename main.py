import os
from torch.multiprocessing import Process
import torch

from utils.args import parse_args
from utils.utils import loggs_to_json, args_to_string
from communication import Peer2PeerNetwork
import time

if __name__ == "__main__":
    torch.manual_seed(1204)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    total_train_time = 0.0
    print("Run experiment in sequential setting..")

    if args.architecture == "ring":
        network = Peer2PeerNetwork(args)
    else:
        print("Not Support yet!")

    start = time.time()
    for k in range(args.n_rounds):
        network.mix()
        if k % 100 == 0:
            end = time.time()
            round_time = (end - start)
            print(f'Round: {k} |Train Time: {round_time:.3f}')
            start = time.time()
    network.write_logs()

    loggs_dir = os.path.join("loggs", args_to_string(args))
    loggs_to_json(loggs_dir)
