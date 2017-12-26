from sklearn.externals import joblib
from sklearn import svm
from os import makedirs, listdir, path
from scipy.cluster.vq import vq
import numpy as np
import cv2
import csv

# Create sift for feature extraction
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

test_data_set_path = "../AID_test/"
_RESULTS_DIR = "Results/"
_COMPARISON_RESULTS_FILENAME = "comparison_results.csv"
_SUMMARY_RESULTS_FILENAME = "summary_results.csv"


def main():
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

    # List where all the descriptors are stored
    des_list = []
    imgs_ref_labels = []
    num_total_images = 0
    for label_folder in listdir(test_data_set_path):
        label_folder_path = path.join(test_data_set_path, label_folder)

        for image_file in listdir(label_folder_path):
            image_path = path.join(label_folder_path, image_file)
            print("read and extracting descriptors: ", image_path)
            # reading image in rgb
            im = cv2.imread(image_path)
            # extracting features
            kpts, des = surf.detectAndCompute(im, None)
            if des is not None:
                imgs_ref_labels += [label_folder];
                des_list.append((image_path, des))
                num_total_images += 1
            else:
                print("Error extracting the descriptor: ", image_path)

    # making the histogram with all the words
    print("Extracting histogram...")
    histogram_with_features = np.zeros((num_total_images, len(centroid_codebook)), 'float32')
    for i in range(num_total_images):
        words, dist = vq(des_list[i][1], centroid_codebook)
        for w in words:
            histogram_with_features[i][w] += 1

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

        all_the_images_tested = [str(img[0]) for img in des_list]
        for i in range(num_total_images):
            comparison_results_writer.writerow([
                all_the_images_tested[i], imgs_ref_labels[i], predictions[i]
            ])

            if imgs_ref_labels[i] == predictions[i]:
                matches += 1
            total += 1

        summary_results_writer.writerow(
            ["match percentage", matches / total * 100])


if __name__ == "__main__":
    main()
