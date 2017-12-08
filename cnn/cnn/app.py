import dataset
import commandline


def main():
    csv_data, dataset_dirpath = commandline.parse()

    annotations_filepath, class_filepath, testing_filepath = dataset.build_cvs_if_none(
        dataset_dirpath, csv_data.dirpath, csv_data.annotations_per_category)

    print(annotations_filepath, class_filepath, testing_filepath)


if __name__ == "__main__":
    main()
