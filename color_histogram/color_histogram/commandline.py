import argparse


class ComputedDataConf():
    def __init__(self, dirpath, nimages_comparison):
        self.dirpath = dirpath
        self.nimages_comparison = nimages_comparison


def parse():
    parser = argparse.ArgumentParser(
        description='A convolution neural network for Aerial scene recognition',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-d',
        '--dataset',
        metavar='dir',
        required=False,
        default="../AID",
        type=str,
        dest="dataset_dirpath",
        help='Directory where the dataset is located')

    parser.add_argument(
        '-cd',
        '--computed-data',
        metavar='dir',
        default="computed_data",
        required=False,
        type=str,
        dest='computed_data_dirpath',
        help=
        'Directory where to place the computed data: histograms, comparison values, statistics, etc.'
    )

    parser.add_argument(
        '-cdn',
        '--computed-data-nimages-comparison',
        metavar='number',
        default=200,
        required=False,
        type=int,
        dest='computed_data_nimages_comparison',
        help=
        'The number of images per category that will serve as a reference for histogram comparison'
    )

    args = parser.parse_args()

    return args.dataset_dirpath, ComputedDataConf(
        args.computed_data_dirpath, args.computed_data_nimages_comparison)
