import argparse
from utils.utils import get_network


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment',
        choices=['driving_carla', 'driving_gazebo'],
        help='name of experiment, possible: driving_carla, driving_gazebo',
        type=str)
    parser.add_argument(
        '--network_name',
        choices=['gaia', 'amazon_us'],
        help='name of the network; possible: gaia, amazon_us',
        type=str
    )
    parser.add_argument(
        '--architecture',
        choices=['ring'],
        help='architecture to use, possible: ring',
        default='ring'
    )
    parser.add_argument(
        '--model',
        choices=['FADNet', 'FADNet_plus'],
        help='model to use, possible: FADNet, FADNet_plus',
        default='FADNet'
    )
    parser.add_argument(
        "--random_ring_proba",
        type=float,
        help="the probability of using a random ring at each step; only used if architecture is ring",
        default=0.5
    )
    parser.add_argument(
        '--fit_by_epoch',
        help='if chosen each local step corresponds to one epoch,'
             ' otherwise each local step corresponds to one gradient step',
        action='store_true'
    )
    parser.add_argument(
        '--n_rounds',
        help='number of communication rounds;',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--bz_train',
        help='batch_size train;',
        type=int,
        default=32
    )
    parser.add_argument(
        '--local_steps',
        help='number of local steps before communication;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--log_freq',
        help='number of local steps before communication;',
        type=int,
        default=100
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or gpu;',
        type=str,
        default='cuda'
    )
    parser.add_argument(
        '--optimizer',
        help='optimizer to be used for the training;',
        type=str,
        default="adam"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help='learning rate',
        default=1e-3
    )
    parser.add_argument(
        "--decay",
        help='learning rate decay scheme to be used;'
             ' possible are "cyclic", "sqrt", "linear" and "constant"(no learning rate decay);'
             'default is "cyclic"',
        type=str,
        default="constant"
    )
    parser.add_argument(
        '--bz_test',
        help='batch_size test;',
        type=int,
        default=1
    )
    parser.add_argument(
        "--test",
        help="if only evaluating test set",
        action='store_true'
    )
    parser.add_argument(
        '--save_logg_path',
        help='path to save logg and models',
        type=str,
        default=""
    )
    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    network = get_network(args.network_name, args.architecture, args.experiment)
    args.num_workers = network.number_of_nodes()

    return args
