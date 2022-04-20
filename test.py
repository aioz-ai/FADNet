from torch.multiprocessing import Process
import torch

from utils.args import parse_args
from communication import Peer2PeerNetwork
if __name__ == "__main__":
    torch.manual_seed(0)
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

    network.load_models(args.n_rounds)
    network.write_logs()
