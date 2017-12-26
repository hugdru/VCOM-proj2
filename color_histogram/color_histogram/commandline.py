import argparse


class ComputedDataConf():
    def __init__(self, dirpath, rebuild_if_exists):
        self.dirpath = dirpath
        self.rebuild_if_exists = rebuild_if_exists


class DatasetConf():
    def __init__(self, train_csv_filepath, test_csv_filepath):
        self.train_csv_filepath = train_csv_filepath
        self.test_csv_filepath = test_csv_filepath


def parse():
    parser = argparse.ArgumentParser(
        description='Color Histogram Comparison for Aerial scene recognition',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-dc',
        '--dataset-csvs',
        metavar='train_csv test_csv',
        required=False,
        default=["../AID_DIVISION/train.csv", "../AID_DIVISION/test.csv"],
        type=_check_tuple_len(2),
        dest="dataset_csvs",
        help='Directory where the dataset csvs are located')

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

    return DatasetConf(
        args.dataset_csvs[0], args.dataset_csvs[1]), ComputedDataConf(
            args.computed_data_dirpath,
            args.rebuild_if_exists), args.images_paths_to_recognize


def _check_tuple_len(n):
    def _(arr):
        return len(arr) != n

    return _
