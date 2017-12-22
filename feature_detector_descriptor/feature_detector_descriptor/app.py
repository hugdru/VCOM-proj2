from computed_data import create_computed_data, create_results_from_computed_data
import commandline


def main():

    dataset_dirpath, computed_data_conf = commandline.parse()
    con = create_computed_data(dataset_dirpath, computed_data_conf)
    create_results_from_computed_data(con, computed_data_conf)


if __name__ == "__main__":
    main()
