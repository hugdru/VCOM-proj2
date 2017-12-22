import csv
import os
import sys
from PIL import Image

LEARNING_ANOTATIONS_FILENAME = "learning_anotations.csv"
LEARNING_CLASS_FILENAME = "learning_class.csv"
TESTING_FILENAME = "testing.csv"


def build_csvs_if_none(dataset_dirpath, csv_data):

    if not os.path.isdir(dataset_dirpath):
        raise Exception(f"{dataset_dirpath} is not a directory")

    os.makedirs(csv_data.dirpath, exist_ok=True)

    learning_annotations_filepath = os.path.join(csv_data.dirpath,
                                                 LEARNING_ANOTATIONS_FILENAME)
    learning_class_filepath = os.path.join(csv_data.dirpath,
                                           LEARNING_CLASS_FILENAME)
    testing_filepath = os.path.join(csv_data.dirpath, TESTING_FILENAME)

    if (not (os.path.isfile(learning_annotations_filepath)
             and os.path.isfile(learning_class_filepath)
             and os.path.isfile(testing_filepath))):
        with open(learning_annotations_filepath,
                  "w") as learning_annotations_file, open(
                      learning_class_filepath,
                      "w") as learning_class_file, open(testing_filepath,
                                                        "w") as testing_file:
            learning_annotations_writer = csv.writer(learning_annotations_file)
            learning_class_writer = csv.writer(learning_class_file)
            testing_writer = csv.writer(testing_file)
            category_id = 0
            for root, _, files in os.walk(dataset_dirpath):
                if not files or os.path.normpath(
                        dataset_dirpath) == os.path.normpath(root):
                    continue
                category = os.path.basename(root).lower()
                learning_class_writer.writerow([category, category_id])
                for i, fn in enumerate(files):
                    fp = os.path.join(root, fn)
                    image_info = Image.open(fp)
                    image_width, image_height = image_info.size
                    if i < csv_data.annotations_per_category:
                        learning_annotations_writer.writerow([
                            fp, 0, 0, image_height - 1, image_width - 1,
                            category
                        ])
                        i += 1
                    else:
                        testing_writer.writerow([fp, category])
                category_id += 1
    else:
        print("csv data files present skipping...", file=sys.stderr)

    return learning_annotations_filepath, learning_class_filepath, testing_filepath
