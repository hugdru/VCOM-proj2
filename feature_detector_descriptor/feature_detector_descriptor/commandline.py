import argparse


class HistogramData:
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
        '-hi',
        '--histogram',
        metavar='dir',
        default="computed_data",
        required=False,
        type=str,
        dest='histogram_data_dirpath',
        help=
        'Directory where to place the data of the histograms. Histogram values of the images for comparison purposes.'
    )

    parser.add_argument(
        '-hinc',
        '--histogram-nimages-comparison',
        metavar='number',
        default=200,
        required=False,
        type=int,
        dest='histogram_nimages_comparison',
        help='The number of images for histogram comparison')

    args = parser.parse_args()

    return HistogramData(
        args.histogram_data_dirpath,
        args.histogram_nimages_comparison), args.dataset_dirpath