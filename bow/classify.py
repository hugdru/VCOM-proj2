from sklearn.externals import joblib
from sklearn import svm
from os import makedirs, path, remove
from scipy.cluster.vq import vq
import numpy as np
import h5py
import cv2
import csv

# Create sift for feature extraction
# cv2.xfeatures2d.SURF_create()
SIFT_SURF = cv2.xfeatures2d.SIFT_create()

# test_data_set_path = "../AID_test/"
_TEST_DATA_SET_CSV = "../AID_DIVISION/test.csv"
_RESULTS_DIR = "Results/"
_COMPARISON_RESULTS_FILENAME = "comparison_results.csv"
_SUMMARY_RESULTS_FILENAME = "summary_results.csv"

_TEMP_DESC_FILE = "descriptors_classify.h5py"

def read_csv(csv_filepath):
    with open(csv_filepath, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            yield row[0], row[1]


def classify():
    makedirs(_RESULTS_DIR, exist_ok=True)
    comparison_results_filepath = path.join(_RESULTS_DIR, _COMPARISON_RESULTS_FILENAME)

    summary_results_filepath = path.join(_RESULTS_DIR, _SUMMARY_RESULTS_FILENAME)

    exists_comparison_results = path.isfile(comparison_results_filepath)
    exists_summary_results = path.isfile(summary_results_filepath)

    if exists_comparison_results and exists_summary_results:
        print("Remove the Results folder")
        exit()

    # load trained data
    print("Loading data in bow.pkl")
    labels, centroid_codebook, clf, stdScaler = joblib.load("bow.pkl")

    descriptor_file_save = h5py.File(_TEMP_DESC_FILE, "w")

    # List where all the descriptors are stored
    imgs_ref_labels = []
    num_total_images = 0
    for path_img, lab in read_csv(_TEST_DATA_SET_CSV):
        print("Extracting descriptors: ", path_img)
        # reading image in rgb
        im = cv2.imread(path_img)
        # extracting features
        kpts, des = SIFT_SURF.detectAndCompute(im, None)
        if des is not None:
            descriptor_file_save.create_dataset(str(num_total_images), data=des)
            imgs_ref_labels.append(lab)
            num_total_images += 1
        else:
            print("Error extracting the descriptor: ", path_img)
    descriptor_file_save.close()

    descriptor_file_load = h5py.File(_TEMP_DESC_FILE, "r")
    # making the histogram with all the words
    print("Extracting histogram...")
    histogram_with_features = np.zeros((num_total_images, len(centroid_codebook)), 'float32')
    for row_id in range(num_total_images):
        words, dist = vq(descriptor_file_load.get(str(row_id)), centroid_codebook)
        for w in words:
            histogram_with_features[row_id][w] += 1
    descriptor_file_load.close()

    # Scale words in the histogram
    print("scale histogram...")
    histogram_with_features = stdScaler.transform(histogram_with_features)

    # get the predictions
    print("Labeling test images...")
    predictions = [labels[i] for i in clf.predict(histogram_with_features)]

    # Check results
    print("Saving Results in files.")
    with open(comparison_results_filepath, "w") as comparison_results_file, open(summary_results_filepath,
                                                                                 "w") as summary_results_file:
        comparison_results_writer = csv.writer(comparison_results_file)
        comparison_results_writer.writerow([
            "testing_image_path", "reference_image_label", "testing_image_label"
        ])

        summary_results_writer = csv.writer(summary_results_file)

        matches = 0
        total = 0

        all_the_images_tested = [str(p) for p, _ in read_csv(_TEST_DATA_SET_CSV)]
        for i in range(num_total_images):
            comparison_results_writer.writerow([
                all_the_images_tested[i], imgs_ref_labels[i], predictions[i]
            ])

            if imgs_ref_labels[i] == predictions[i]:
                matches += 1
            total += 1

        summary_results_writer.writerow(
            ["match percentage", matches / total * 100])
    remove(_TEMP_DESC_FILE)


if __name__ == "__main__":
    classify()
