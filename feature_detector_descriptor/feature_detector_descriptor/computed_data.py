import os
import sqlite3
import pickle
import cv2 as cv

_COMPUTED_DATA_FILENAME = "computed_data.sqlite3"


def create_computed_data(dataset_dirpath,
                         computed_data_conf,
                         rebuild_if_exists=False):

    if not os.path.isdir(dataset_dirpath):
        raise Exception(f"{dataset_dirpath} is not a directory")

    os.makedirs(computed_data_conf.dirpath, exist_ok=True)

    computed_data_filepath = os.path.join(computed_data_conf.dirpath,
                                          _COMPUTED_DATA_FILENAME)

    exists_computed_data = os.path.isfile(computed_data_filepath)

    if exists_computed_data and not rebuild_if_exists:
        return _create_sqlite3_file(computed_data_filepath)

    con = _create_sqlite3_file(computed_data_filepath, remove_if_exists=True)

    _create_tables(con)
    _create_histogram_data(con, dataset_dirpath, computed_data_conf)
    _create_histogram_comparison_data(con)


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
        for f_index, f in enumerate(files):
            if f_index < computed_data_conf.nimages_comparison:
                image_type = Histogram.REFERENCE_IMAGE_TYPE
            else:
                image_type = Histogram.TESTING_IMAGE_TYPE

            image_path = os.path.join(root, f)
            image = cv.imread(image_path)
            bucket_size = 32
            histogram_data = cv.calcHist([image], [0, 1, 2], None,
                                         [bucket_size] * 3,
                                         [0, 256, 0, 256, 0, 256])

            computed_histogram = Histogram(
                image_path=image_path,
                label=label,
                histogram_data=histogram_data,
                bucket_size=bucket_size,
                image_type=image_type)

            computed_histogram.insert(con)
    con.commit()


def _create_histogram_comparison_data(con):

    reference_images_iter = Histogram.select_filtered_iter(
        con, {
            "image_type": Histogram.REFERENCE_IMAGE_TYPE
        })

    testing_images_iter = Histogram.select_filtered_iter(
        con, {
            "image_type": Histogram.TESTING_IMAGE_TYPE
        })

    for reference_image in reference_images_iter:
        for testing_image in testing_images_iter:
            comparison_results = HistogramComparison(
                reference_image.id_pk, testing_image.id_pk,
                HistogramComparison.DISTANCE_METHOD,
                cv.compareHist(reference_image.histogram_data,
                               testing_image.histogram_data,
                               HistogramComparison.DISTANCE_METHOD))
            comparison_results.insert(con)

    con.commit()


class Histogram():

    TABLE_NAME = "histogram"
    REFERENCE_IMAGE_TYPE = 1
    TESTING_IMAGE_TYPE = 2

    def __init__(self,
                 image_path,
                 label,
                 histogram_data,
                 bucket_size,
                 image_type,
                 id_pk=None):
        self.image_path = image_path
        self.label = label
        self.histogram_data = histogram_data
        self.bucket_size = bucket_size
        self.image_type = image_type
        self.id_pk = id_pk

    def insert(self, con):
        con.execute(
            "INSERT INTO {}(image_path, label, bucket_size, histogram_data, image_type) VALUES (?, ?, ?, ?, ?)".
            format(self.TABLE_NAME), [
                self.image_path, self.label, self.bucket_size,
                pickle.dumps(self.histogram_data), self.image_type
            ])

    @classmethod
    def _create_from_row(cls, row):
        return cls(
            id_pk=row["id"],
            image_path=row["image_path"],
            label=row["label"],
            histogram_data=pickle.loads(row["histogram_data"]),
            bucket_size=row["bucket_size"],
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
            where_str, values = build_where_str_from_filter(where_filter)
            if not where_str:
                return False

            for row in cursor.execute("SELECT * FROM {} WHERE {}".format(
                    cls.TABLE_NAME, where_str), values):
                yield cls._create_from_row(row)
        finally:
            cursor.close()

        return True

    @classmethod
    def create_table(cls, con):
        con.execute(
            "CREATE TABLE {} (id INTEGER PRIMARY KEY AUTOINCREMENT, image_path TEXT UNIQUE NOT NULL, label TEXT NOT NULL, bucket_size INTEGER NOT NULL, histogram_data BLOB NOT NULL, image_type INTEGER NOT NULL)".
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

    def insert(self, con):
        con.execute(
            "INSERT INTO {}(reference_image_id, testing_image_id, distance_method, distance) VALUES (?, ?, ?, ?)".
            format(self.TABLE_NAME), [
                self.reference_image_id, self.testing_image_id,
                self.distance_method, self.distance
            ])


def build_where_str_from_filter(where_filter):

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
