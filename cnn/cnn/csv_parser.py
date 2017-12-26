import sys

import dataset
import commandline


def main():
    csv_data, dataset_dirpath = commandline.parse()

    annotations_filepath, class_filepath, testing_filepath = dataset.build_csvs_if_none(
        dataset_dirpath, csv_data)

    print(
        dataset_dirpath,
        annotations_filepath,
        class_filepath,
        testing_filepath,
        file=sys.stderr)


if __name__ == "__main__":
    main()
