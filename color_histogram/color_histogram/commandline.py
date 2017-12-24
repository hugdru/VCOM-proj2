import argparse


class ComputedDataConf():
    def __init__(self, dirpath, nimages_comparison, rebuild_if_exists):
        self.dirpath = dirpath
        self.nimages_comparison = nimages_comparison
        self.rebuild_if_exists = rebuild_if_exists


def parse():
    parser = argparse.ArgumentParser(
        description='Color Histogram Comparison for Aerial scene recognition',
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

    parser.add_argument(
        '-ri',
        '--recognize-images',
        metavar='file',
        required=False,
        type=str,
        nargs='+',
        dest='images_paths_to_recognize',
        help='External images to be recognized')

    parser.add_argument(
        "-f",
        "--force",
        required=False,
        default=False,
        action='store_true',
        dest="rebuild_if_exists",
        help="force reconstruction of computed_data")

    args = parser.parse_args()

    return args.dataset_dirpath, ComputedDataConf(
        args.computed_data_dirpath, args.computed_data_nimages_comparison,
        args.rebuild_if_exists), args.images_paths_to_recognize
