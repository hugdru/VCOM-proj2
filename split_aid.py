from os import path, makedirs, walk
from shutil import move
from glob import glob
import random
import math
import csv

# 200 para test
dataset_dirpath = "AID/"

train_dataset_folder = "AID_DIVISION/"
test_dataset_folder = "AID_DIVISION/"
train_csv_file = "train.csv"
test_csv_file = "test.csv"


def separate(nimages_comparison):
    makedirs(train_dataset_folder, exist_ok=True)
    makedirs(test_dataset_folder, exist_ok=True)
    train_csv_filepath = path.join(train_dataset_folder, train_csv_file)
    test_csv_filepath = path.join(test_dataset_folder, test_csv_file)

    with open(train_csv_filepath, "w") as train_file, open(test_csv_filepath,
                                                           "w") as test_file:
        train_writer = csv.writer(train_file)
        train_writer.writerow(["image_path", "image_label"])

        test_writer = csv.writer(test_file)
        test_writer.writerow(["image_path", "image_label"])

        for root, _, files in walk(dataset_dirpath):
            if not files or path.normpath(dataset_dirpath) == path.normpath(
                    root):
                continue
            label = path.basename(root).lower()
            for f_index, partial_fpath in enumerate(files):
                image_path = path.join(root, partial_fpath)
                if f_index < nimages_comparison:
                    train_writer.writerow([image_path, label])
                else:
                    test_writer.writerow([image_path, label])


def read_csv(csv_filepath):
    with open(csv_filepath, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            yield row[0], row[1]


if __name__ == "__main__":
    separate(200)
