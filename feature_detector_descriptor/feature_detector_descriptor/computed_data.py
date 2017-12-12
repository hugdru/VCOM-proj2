import json
import os
import cv2 as cv

HISTOGRAM_FILENAME = "histogram.json"


class ComputedData():
    def __init__(self):
        self.computed_histograms = []
        self.testing_images = []


class ComputedHistogram():
    def __init__(self, image_path, label, histogram, bucket_size):
        self.image_path = image_path
        self.label = label
        self.histogram = histogram
        self.bucket_size = bucket_size


class TestingImage():
    def __init__(self, image_path, label):
        self.image_path = image_path
        self.label = label


def build_computed_data_if_none(dataset_dirpath, histogram_data):

    if not os.path.isdir(dataset_dirpath):
        raise Exception(f"{dataset_dirpath} is not a directory")

    os.makedirs(histogram_data.dirpath, exist_ok=True)

    histogram_filepath = os.path.join(histogram_data.dirpath,
                                      HISTOGRAM_FILENAME)
    computed_data = None
    if not os.path.isfile(histogram_filepath):
        computed_data = ComputedData()
        with open(histogram_filepath, "w") as histogram_file:
            for root, _, files in os.walk(dataset_dirpath):
                if not files or os.path.normpath(
                        dataset_dirpath) == os.path.normpath(root):
                    continue
                label = os.path.basename(root).lower()
                for f_index, f in enumerate(files):
                    if f_index < histogram_data.nimages_comparison:
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
                            bucket_size=bucket_size)
                        computed_data.computed_histograms.append(
                            computed_histogram)
                    else:
                        testing_image = TestingImage(
                            image_path=image_path, label=label)
                        computed_data.testing_images.append(testing_image)
            computed_data_serialize = json.dumps(
                computed_data, default=lambda o: o.__dict__)
            histogram_file.write(computed_data_serialize)

    else:
        # deseralized the file
        pass

    return computed_data
