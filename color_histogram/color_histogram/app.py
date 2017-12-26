from computed_data import create_computed_data, create_results_from_computed_data, recognize_images
import commandline


def main():

    dataset_conf, computed_data_conf, images_paths_to_recognize = commandline.parse(
    )

    con = create_computed_data(dataset_conf, computed_data_conf)
    create_results_from_computed_data(con, computed_data_conf)
    recognize_images(con, images_paths_to_recognize)


if __name__ == "__main__":
    main()
