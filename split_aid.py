from os import path, makedirs, walk
import csv

DATASET_DIRPATH = "AID/"
SPLIT_AID_FOLDER = "AID_DIVISION/"
TRAIN_CSV_FILENAME = "train.csv"
TEST_CSV_FILENAME = "test.csv"
NIMAGES_PER_CATEGORY = 200


def create_csvs():

    makedirs(SPLIT_AID_FOLDER, exist_ok=True)

    train_csv_filepath = path.join(SPLIT_AID_FOLDER, TRAIN_CSV_FILENAME)
    test_csv_filepath = path.join(SPLIT_AID_FOLDER, TEST_CSV_FILENAME)

    with open(train_csv_filepath, "w") as train_file, open(
            test_csv_filepath, "w") as test_file:

        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)

        for root, _, files in walk(DATASET_DIRPATH):
            if not files or path.normpath(DATASET_DIRPATH) == path.normpath(
                    root):
                continue
            label = path.basename(root).lower()
            for f_index, partial_fpath in enumerate(files):
                image_path = path.join("../", root, partial_fpath)
                if f_index < NIMAGES_PER_CATEGORY:
                    train_writer.writerow([image_path, label])
                else:
                    test_writer.writerow([image_path, label])


def read_csv(csv_filepath):
    with open(csv_filepath, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            yield row[0], row[1]


if __name__ == "__main__":
    create_csvs()
