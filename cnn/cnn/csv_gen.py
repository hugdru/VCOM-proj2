import sys

import dataset
import commandline


def main():
    dataset_conf, csv_data = commandline.parse()

    dataset.build_csvs_if_none(dataset_conf, csv_data)


if __name__ == "__main__":
    main()
