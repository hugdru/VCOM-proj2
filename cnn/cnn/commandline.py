import argparse


class CsvData:
    def __init__(self, dirpath, annotations_per_category):
        self.dirpath = dirpath
        self.annotations_per_category = annotations_per_category


class DatasetConf():
    def __init__(self, train_csv_filepath, test_csv_filepath,
                 labels_csv_filepath):
        self.train_csv_filepath = train_csv_filepath
        self.test_csv_filepath = test_csv_filepath
        self.labels_csv_filepath = labels_csv_filepath


def parse():
    parser = argparse.ArgumentParser(
        description='A convolution neural network for Aerial scene recognition',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-dc',
        '--dataset-csvs',
        metavar='train_csv test_csv labels_csv',
        required=False,
        default=[
            "../AID_DIVISION/train.csv", "../AID_DIVISION/test.csv",
            "../AID_DIVISION/labels.csv"
        ],
        type=_check_tuple_len(3),
        dest="dataset_csvs",
        help='Directory where the dataset csvs are located')

    parser.add_argument(
        '-c',
        '--csv',
        metavar='dir',
        default="csv_data",
        required=False,
        type=str,
        dest='csv_data_dirpath',
        help=
        'Directory where to place the csv data, references to the learning and testing images.'
    )

    parser.add_argument(
        '-cp',
        '--csv-per-category',
        metavar='number',
        default=200,
        required=False,
        type=int,
        dest='csv_data_annotations_per_category',
        help='The number of images per category for learning')

    args = parser.parse_args()

    return DatasetConf(args.dataset_csvs[0],
                       args.dataset_csvs[1], args.dataset_csvs[2]), CsvData(
                           args.csv_data_dirpath,
                           args.csv_data_annotations_per_category)


def _check_tuple_len(n):
    def _(arr):
        return len(arr) != n

    return _