from computed_data import create_computed_data, Histogram
import commandline


def main():

    dataset_dirpath, computed_data_conf = commandline.parse()
    create_computed_data(dataset_dirpath, computed_data_conf)


if __name__ == "__main__":
    main()
