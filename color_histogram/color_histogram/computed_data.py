import os
import sqlite3
import pickle
import csv
import cv2 as cv

_COMPUTED_DATA_FILENAME = "computed_data.sqlite3"
_COMPARISON_RESULTS_FILENAME = "comparison_results.csv"
_SUMMARY_RESULTS_FILENAME = "summary_results.csv"


def create_computed_data(dataset_dirpath, computed_data_conf):

    os.makedirs(computed_data_conf.dirpath, exist_ok=True)

    computed_data_filepath = os.path.join(computed_data_conf.dirpath,
                                          _COMPUTED_DATA_FILENAME)

    exists_computed_data = os.path.isfile(computed_data_filepath)

    if exists_computed_data and not computed_data_conf.rebuild_if_exists:
        return _create_sqlite3_file(computed_data_filepath)

    if not os.path.isdir(dataset_dirpath):
        raise Exception(f"{dataset_dirpath} is not a directory")

    con = _create_sqlite3_file(computed_data_filepath, remove_if_exists=True)

    _create_tables(con)
    _create_histogram_data(con, dataset_dirpath, computed_data_conf)
    _create_histogram_comparison_data(con)

    return con


def create_results_from_computed_data(con, computed_data_conf):

    comparison_results_filepath = os.path.join(computed_data_conf.dirpath,
                                               _COMPARISON_RESULTS_FILENAME)

    summary_results_filepath = os.path.join(computed_data_conf.dirpath,
                                            _SUMMARY_RESULTS_FILENAME)

    exists_comparison_results = os.path.isfile(comparison_results_filepath)
    exists_summary_results = os.path.isfile(summary_results_filepath)

    if exists_comparison_results and exists_summary_results and not computed_data_conf.rebuild_if_exists:
        return

    cursor = con.cursor()
    try:
        # The outer group by is used to pick just one
        # reference image per testing image, min(o.reference_image_id).
        # Another aggregate function could be used. Leading to different
        # matching percentages. Maybe a random one would be interesting.
        # This can happen because there might be reference_image_ids
        # or distance_methods with the same testing_image_id and distance.
        query = """
SELECT cr.reference_image_id, cr.testing_image_id, hr.label AS reference_image_label, ht.label AS testing_image_label
FROM (
  SELECT MIN(o.reference_image_id) AS reference_image_id, i.testing_image_id
  FROM histogram_comparison o
  INNER JOIN (
    SELECT testing_image_id, MIN(distance) AS distance
    FROM histogram_comparison
    GROUP BY testing_image_id) i
  ON
    o.testing_image_id = i.testing_image_id AND
    o.distance = i.distance
  GROUP BY i.testing_image_id) cr
INNER JOIN histogram hr ON hr.id = cr.reference_image_id
INNER JOIN histogram ht ON ht.id = cr.testing_image_id"""

        with open(comparison_results_filepath,
                  "w") as comparison_results_file, open(
                      summary_results_filepath, "w") as summary_results_file:
            comparison_results_writer = csv.writer(comparison_results_file)
            comparison_results_writer.writerow([
                "reference_image_id", "testing_image_id",
                "reference_image_label", "testing_image_label"
            ])

            summary_results_writer = csv.writer(summary_results_file)

            matches = 0
            total = 0

            for row in cursor.execute(query):
                if row["reference_image_label"] == row["testing_image_label"]:
                    matches += 1

                comparison_results_writer.writerow([
                    row["reference_image_id"], row["testing_image_id"],
                    row["reference_image_label"], row["testing_image_label"]
                ])

                total += 1

            summary_results_writer.writerow(
                ["match percentage", matches / total * 100])

    finally:
        cursor.close()


def recognize_images(con, images_paths_to_recognize):
    if not images_paths_to_recognize:
        return

    for image_path_to_recognize in images_paths_to_recognize:
        image_histogram_to_recognize = _create_image_histogram(
            cv.imread(image_path_to_recognize))

        selected_reference_image = None
        selected_distance = None
        for reference_image in Histogram.select_reference_type(con):
            distance = _compare_histograms(reference_image.histogram_data,
                                           image_histogram_to_recognize)
            if not selected_distance or selected_distance > distance:
                selected_reference_image = reference_image
                selected_distance = distance

        if selected_reference_image:
            print("{} is a {}".format(image_path_to_recognize,
                                      selected_reference_image.label))
        else:
            raise Exception("No reference images present in the database")


def _create_sqlite3_file(filepath, remove_if_exists=False):

    if remove_if_exists:
        try:
            os.remove(filepath)
        except OSError:
            pass

    con = sqlite3.connect(filepath)
    con.row_factory = sqlite3.Row
    return con


def _create_tables(con):
    Histogram.create_table(con)
    HistogramComparison.create_table(con)
    con.commit()


def _create_histogram_data(con, dataset_dirpath, computed_data_conf):

    for root, _, files in os.walk(dataset_dirpath):
        if not files or os.path.normpath(dataset_dirpath) == os.path.normpath(
                root):
            continue
        label = os.path.basename(root).lower()
        for f_index, partial_fpath in enumerate(files):
            if f_index < computed_data_conf.nimages_comparison:
                image_type = Histogram.REFERENCE_IMAGE_TYPE
            else:
                image_type = Histogram.TESTING_IMAGE_TYPE

            image_path = os.path.join(root, partial_fpath)

            computed_histogram = Histogram.build_load_image(
                image_path=image_path, label=label, image_type=image_type)

            computed_histogram.insert(con)
    con.commit()


def _create_image_histogram(image):
    bucket_size = 32
    return cv.calcHist([image], [0, 1, 2], None, [bucket_size] * 3,
                       [0, 256, 0, 256, 0, 256])


def _create_histogram_comparison_data(con):

    reference_images_iter = Histogram.select_reference_type(con)

    for reference_image in reference_images_iter:

        testing_images_iter = Histogram.select_testing_type(con)

        for testing_image in testing_images_iter:
            distance_method, distance = _compare_histograms(
                reference_image.histogram_data, testing_image.histogram_data)

            histogram_comparison = HistogramComparison(
                reference_image_id=reference_image.id_pk,
                testing_image_id=testing_image.id_pk,
                distance_method=distance_method,
                distance=distance)
            histogram_comparison.insert(con)

    con.commit()


def _compare_histograms(reference_image_histogram, testing_image_histogram):
    return HistogramComparison.DISTANCE_METHOD, cv.compareHist(
        reference_image_histogram, testing_image_histogram,
        HistogramComparison.DISTANCE_METHOD)


class Histogram():

    TABLE_NAME = "histogram"
    REFERENCE_IMAGE_TYPE = 1
    TESTING_IMAGE_TYPE = 2

    def __init__(self,
                 image_path,
                 label,
                 histogram_data,
                 image_type,
                 id_pk=None):
        self.image_path = image_path
        self.label = label
        self.histogram_data = histogram_data
        self.image_type = image_type
        self.id_pk = id_pk

    @classmethod
    def build_load_image(cls, image_path, label, image_type):
        return cls(
            image_path=image_path,
            label=label,
            histogram_data=_create_image_histogram(cv.imread(image_path)),
            image_type=image_type)

    def insert(self, con):
        con.execute(
            "INSERT INTO {}(image_path, label, histogram_data, image_type) VALUES (?, ?, ?, ?)".
            format(self.TABLE_NAME), [
                self.image_path, self.label,
                pickle.dumps(self.histogram_data), self.image_type
            ])

    @classmethod
    def _create_from_row(cls, row):
        return cls(
            id_pk=row["id"],
            image_path=row["image_path"],
            label=row["label"],
            histogram_data=pickle.loads(row["histogram_data"]),
            image_type=row["image_type"])

    @classmethod
    def select_all_iter(cls, con):
        cursor = con.cursor()
        try:
            for row in cursor.execute("SELECT * FROM {}".format(
                    cls.TABLE_NAME)):
                yield cls._create_from_row(row)
        finally:
            cursor.close()

        return True

    @classmethod
    def select_filtered_iter(cls, con, where_filter):
        cursor = con.cursor()
        try:
            where_str, values = _build_where_str_from_filter(where_filter)
            if not where_str:
                return False

            for row in cursor.execute("SELECT * FROM {} WHERE {}".format(
                    cls.TABLE_NAME, where_str), values):
                yield cls._create_from_row(row)
        finally:
            cursor.close()

        return True

    @classmethod
    def select_reference_type(cls, con):
        return Histogram.select_filtered_iter(
            con, {
                "image_type": Histogram.REFERENCE_IMAGE_TYPE
            })

    @classmethod
    def select_testing_type(cls, con):
        return Histogram.select_filtered_iter(
            con, {
                "image_type": Histogram.TESTING_IMAGE_TYPE
            })

    @classmethod
    def create_table(cls, con):
        con.execute(
            "CREATE TABLE {} (id INTEGER PRIMARY KEY AUTOINCREMENT, image_path TEXT UNIQUE NOT NULL, label TEXT NOT NULL, histogram_data BLOB NOT NULL, image_type INTEGER NOT NULL)".
            format(cls.TABLE_NAME))

        con.execute(
            "CREATE INDEX {0}_image_type_idx ON {0}(image_type)".format(
                cls.TABLE_NAME))


class HistogramComparison():

    DISTANCE_METHOD = cv.HISTCMP_HELLINGER
    TABLE_NAME = "histogram_comparison"

    def __init__(self, reference_image_id, testing_image_id, distance_method,
                 distance):
        self.reference_image_id = reference_image_id
        self.testing_image_id = testing_image_id
        self.distance_method = distance_method
        self.distance = distance

    @classmethod
    def create_table(cls, con):
        con.execute("""CREATE TABLE {0} (
reference_image_id INTEGER NOT NULL,
testing_image_id INTEGER NOT NULL,
distance_method INTEGER NOT NULL,
distance REAL NOT NULL,
FOREIGN KEY (reference_image_id) REFERENCES {1} (id),
FOREIGN KEY (testing_image_id) REFERENCES {1} (id),
PRIMARY KEY (reference_image_id, testing_image_id, distance_method))""".format(
            cls.TABLE_NAME, Histogram.TABLE_NAME))

        con.execute(
            "CREATE INDEX {0}_statistics_idx ON {0}(testing_image_id, distance, reference_image_id)".
            format(cls.TABLE_NAME))

    def insert(self, con):
        con.execute(
            "INSERT INTO {}(reference_image_id, testing_image_id, distance_method, distance) VALUES (?, ?, ?, ?)".
            format(self.TABLE_NAME), [
                self.reference_image_id, self.testing_image_id,
                self.distance_method, self.distance
            ])


def _build_where_str_from_filter(where_filter):

    if not where_filter:
        return None, None

    filter_iter = iter(where_filter.items())

    try:
        first_column, first_value = next(filter_iter)
    except StopIteration:
        return None, None

    where_str = " {} = ? ".format(first_column)
    values = [first_value]

    for column, value in filter_iter:
        where_str = where_str + "AND {} = ? ".format(column)
        values.append(value)

    return where_str, tuple(values)
