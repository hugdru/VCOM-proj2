import os
import sqlite3
import cv2 as cv
import marshal

HISTOGRAM_FILENAME = "histogram.sqlite3"
COMPARING_IMAGE_TYPE = 1
TESTING_IMAGE_TYPE = 2


class ComputedHistogram():
    def __init__(self, image_path, label, histogram, bucket_size, image_type):
        self.image_path = image_path
        self.label = label
        self.histogram = histogram
        self.bucket_size = bucket_size
        self.image_type = image_type

    def insert(self, con):
        con.execute(
            "INSERT INTO computed_histogram(image_path, label, bucket_size, histogram, image_type) VALUES (?, ?, ?, ?, ?)",
            [
                self.image_path, self.label, self.bucket_size,
                marshal.dumps(self.histogram), self.image_type
            ])


def build_computed_data_if_none(dataset_dirpath, histogram_data):
    if not os.path.isdir(dataset_dirpath):
        raise Exception(f"{dataset_dirpath} is not a directory")

    os.makedirs(histogram_data.dirpath, exist_ok=True)

    histogram_filepath = os.path.join(histogram_data.dirpath,
                                      HISTOGRAM_FILENAME)

    exists_histogram_filepath = os.path.isfile(histogram_filepath)
    con = sqlite3.connect(histogram_filepath)

    if not exists_histogram_filepath:

        _create_db(con)

        for root, _, files in os.walk(dataset_dirpath):
            if not files or os.path.normpath(
                    dataset_dirpath) == os.path.normpath(root):
                continue
            label = os.path.basename(root).lower()
            for f_index, f in enumerate(files):
                if f_index < histogram_data.nimages_comparison:
                    image_type = COMPARING_IMAGE_TYPE
                else:
                    image_type = TESTING_IMAGE_TYPE

                image_path = os.path.join(root, f)
                image = cv.imread(image_path)
                bucket_size = 32
                histogram = cv.calcHist([image], [0, 1, 2], None,
                                        [bucket_size] * 3,
                                        [0, 256, 0, 256, 0, 256])

                computed_histogram = ComputedHistogram(
                    image_path=image_path,
                    label=label,
                    histogram=histogram,
                    bucket_size=bucket_size,
                    image_type=image_type)

                computed_histogram.insert(con)
        con.commit()

    return con


def _create_db(con):
    con.execute(
        "CREATE TABLE IF NOT EXISTS computed_histogram (id INTEGER PRIMARY KEY AUTOINCREMENT, image_path TEXT UNIQUE NOT NULL, label TEXT NOT NULL, bucket_size INTEGER NOT NULL, histogram BLOB NOT NULL, image_type INTEGER NOT NULL)"
    )
    con.execute(
        "CREATE INDEX computed_histogram_image_type_idx ON computed_histogram(image_type)"
    )
    con.commit()
