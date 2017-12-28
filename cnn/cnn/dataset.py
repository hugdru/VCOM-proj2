import csv
import os
import sys
from PIL import Image

LEARNING_ANNOTATIONS_FILENAME = "learning_annotations.csv"
CLASSES_FILENAME = "classes.csv"
TESTING_ANNOTATIONS_FILENAME = "testing_annotations.csv"


def build_csvs_if_none(dataset_conf, csv_data):

    os.makedirs(csv_data.dirpath, exist_ok=True)

    learning_annotations_filepath = os.path.join(csv_data.dirpath,
                                                 LEARNING_ANNOTATIONS_FILENAME)
    learning_class_filepath = os.path.join(csv_data.dirpath, CLASSES_FILENAME)
    testing_annotations_filepath = os.path.join(csv_data.dirpath,
                                                TESTING_ANNOTATIONS_FILENAME)

    if (os.path.isfile(learning_annotations_filepath)
            and os.path.isfile(learning_class_filepath)
            and os.path.isfile(testing_annotations_filepath)):
        print("csv data files present skipping...", file=sys.stderr)
        return

    if not (os.path.isfile(dataset_conf.train_csv_filepath)
            and os.path.isfile(dataset_conf.test_csv_filepath)
            and os.path.isfile(dataset_conf.labels_csv_filepath)):
        raise Exception(f"dataset csvs do not exist")

    with open(learning_annotations_filepath,
              "w") as learning_annotations_file, open(
                  learning_class_filepath, "w") as classes_file, open(
                      testing_annotations_filepath,
                      "w") as testing_annotations_file:
        learning_annotations_writer = csv.writer(learning_annotations_file)
        testing_annotations_writer = csv.writer(testing_annotations_file)
        classes_writer = csv.writer(classes_file)

        _csv_train_test_writer(learning_annotations_writer,
                               _read_csv(dataset_conf.train_csv_filepath))

        _csv_train_test_writer(testing_annotations_writer,
                               _read_csv(dataset_conf.test_csv_filepath))

        labels_iterator = _read_csv(dataset_conf.labels_csv_filepath)

        for i, image_class in enumerate(labels_iterator):
            classes_writer.writerow([image_class[0], i])


def _csv_train_test_writer(writer, csv_it):
    for image_path, image_label in csv_it:
        image_width, image_height = _get_image_size(image_path)
        writer.writerow(
            [image_path, 0, 0, image_height - 1, image_width - 1, image_label])


def _get_image_size(image_path):
    image_info = Image.open(image_path)
    return image_info.size


def _read_csv(csv_filepath):
    with open(csv_filepath, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            yield row
