import csv
import os
import sys
from PIL import Image

LEARNING_ANOTATIONS_FILENAME = "learning_anotations.csv"
LEARNING_CLASS_FILENAME = "learning_class.csv"
TESTING_FILENAME = "testing.csv"


def build_csvs_if_none(dataset_conf, csv_data):

    os.makedirs(csv_data.dirpath, exist_ok=True)

    learning_annotations_filepath = os.path.join(csv_data.dirpath,
                                                 LEARNING_ANOTATIONS_FILENAME)
    learning_class_filepath = os.path.join(csv_data.dirpath,
                                           LEARNING_CLASS_FILENAME)
    testing_filepath = os.path.join(csv_data.dirpath, TESTING_FILENAME)

    if (os.path.isfile(learning_annotations_filepath)
            and os.path.isfile(learning_class_filepath)
            and os.path.isfile(testing_filepath)):
        print("csv data files present skipping...", file=sys.stderr)
        return

    if not (os.path.isfile(dataset_conf.train_csv_filepath)
            and os.path.isfile(dataset_conf.test_csv_filepath)
            and os.path.isfile(dataset_conf.labels_csv_filepath)):
        raise Exception(f"dataset csvs do not exist")

    with open(learning_annotations_filepath,
              "w") as learning_annotations_file, open(
                  learning_class_filepath, "w") as learning_class_file, open(
                      testing_filepath, "w") as testing_file:
        learning_annotations_writer = csv.writer(learning_annotations_file)
        testing_writer = csv.writer(testing_file)
        learning_class_writer = csv.writer(learning_class_file)

        train_iterator = _read_csv(dataset_conf.train_csv_filepath)

        for image_path, image_label in train_iterator:
            image_width, image_height = _get_image_size(image_path)
            learning_annotations_writer.writerow([
                image_path, 0, 0, image_height - 1, image_width - 1,
                image_label
            ])

        test_iterator = _read_csv(dataset_conf.test_csv_filepath)

        for image_path, image_label in test_iterator:
            image_width, image_height = _get_image_size(image_path)
            testing_writer.writerow([image_path, image_label])

        labels_iterator = _read_csv(dataset_conf.labels_csv_filepath)

        for i, image_class in enumerate(labels_iterator):
            learning_class_writer.writerow([image_class[0], i])


def _get_image_size(image_path):
    image_info = Image.open(image_path)
    return image_info.size


def _read_csv(csv_filepath):
    with open(csv_filepath, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            yield row
