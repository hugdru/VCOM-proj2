from os import path, makedirs, walk
import csv

DATASET_DIRPATH = "AID/"
SPLIT_AID_FOLDER = "AID_DIVISION/"
TRAIN_CSV_FILENAME = "train.csv"
TEST_CSV_FILENAME = "test.csv"
LABELS_CSV_FILENAME = "labels.csv"
NIMAGES_PER_CATEGORY = 200


def create_csvs():

    makedirs(SPLIT_AID_FOLDER, exist_ok=True)

    train_csv_filepath = path.join(SPLIT_AID_FOLDER, TRAIN_CSV_FILENAME)
    test_csv_filepath = path.join(SPLIT_AID_FOLDER, TEST_CSV_FILENAME)
    labels_csv_filepath = path.join(SPLIT_AID_FOLDER, LABELS_CSV_FILENAME)

    exists_train_csv = path.isfile(train_csv_filepath)
    exists_test_csv = path.isfile(test_csv_filepath)
    exists_labels_csv = path.isfile(labels_csv_filepath)

    if exists_train_csv and exists_test_csv and exists_labels_csv:
        return

    with open(train_csv_filepath, "w") as train_file, open(
            test_csv_filepath, "w") as test_file, open(labels_csv_filepath,
                                                       "w") as labels_file:

        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)
        labels_writer = csv.writer(labels_file)

        for root, _, files in walk(DATASET_DIRPATH):
            if not files or path.normpath(DATASET_DIRPATH) == path.normpath(
                    root):
                continue
            label = path.basename(root).lower()
            labels_writer.writerow([label])
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
            yield row


if __name__ == "__main__":
    create_csvs()
