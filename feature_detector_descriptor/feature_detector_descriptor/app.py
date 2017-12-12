import computed_data
import commandline


def main():

    histogram_data, dataset_dirpath = commandline.parse()
    computed_data.build_computed_data_if_none(dataset_dirpath, histogram_data)


if __name__ == "__main__":
    main()
