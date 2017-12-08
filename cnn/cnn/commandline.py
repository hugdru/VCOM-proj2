import argparse


class CsvData:
    def __init__(self, dirpath, annotations_per_category):
        self.dirpath = dirpath
        self.annotations_per_category = annotations_per_category


def parse():
    parser = argparse.ArgumentParser(
        description='A convolution neural network for Aerial scene recognition',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-d',
        '--dataset',
        metavar='dir',
        required=False,
        default="AID",
        type=str,
        dest="dataset_dirpath",
        help='Directory where the dataset is located')

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

    return CsvData(
        args.csv_data_dirpath,
        args.csv_data_annotations_per_category), args.dataset_dirpath
