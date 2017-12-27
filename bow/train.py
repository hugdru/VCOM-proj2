from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import svm
from scipy.cluster.vq import vq, kmeans
from os import remove
import csv
import numpy as np
import cv2
import h5py

# Create sift for feature extraction
# cv2.xfeatures2d.SURF_create()
SIFT_SURF = cv2.xfeatures2d.SIFT_create()
STACK_COLS = 128

# path to aid folder
_TRAIN_DATA_SET_SCV = "../AID_DIVISION/train.csv"
_TEMP_DESC_FILE = "descriptors_train.h5py"
_TEMP_DESC_STACK_FILE = "descriptors_train_stacking.h5py"


def read_csv(csv_filepath):
    with open(csv_filepath, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            yield row[0], row[1]


def train():
    # get all the folders inside the aid folder(classes)
    descriptor_file_save = h5py.File(_TEMP_DESC_FILE, "w")

    num_total_images = 0
    possible_labels_folders = []
    image_label_list = []
    label_id = -1
    num_rows_des = 0
    for path_img, lab in read_csv(_TRAIN_DATA_SET_SCV):
        print("Extracting descriptors: ", path_img)
        # reading image in rgb
        im = cv2.imread(path_img)
        # extracting features
        kpts, des = SIFT_SURF.detectAndCompute(im, None)
        if des is not None:
            if lab not in possible_labels_folders:
                label_id += 1
                print("Changing label (", lab, " - ", label_id, "): ", path_img)
                possible_labels_folders.append(lab)

            num_rows_des += len(des)
            descriptor_file_save.create_dataset(str(num_total_images), data=des)
            num_total_images += 1
            image_label_list.append(label_id)
        else:
            print("Error extracting the descriptor: ", path_img)
    descriptor_file_save.close()

    descriptor_file_load = h5py.File(_TEMP_DESC_FILE, "r")

    stacking_descriptor_save = h5py.File(_TEMP_DESC_STACK_FILE, 'w')
    stack = stacking_descriptor_save.create_dataset('stack_kmeans',
                                                    (num_rows_des, STACK_COLS), maxshape=(None, None))
    row_pointer = 0
    for row_id in range(num_total_images):
        desc = descriptor_file_load.get(str(row_id))
        size_desc = len(desc)
        end_value = row_pointer + len(desc)
        print(row_id, " - ", size_desc, " -> ", row_pointer, "/", end_value, "/", num_rows_des)
        stack[row_pointer:end_value, 0:STACK_COLS] = desc
        row_pointer = end_value

    stacking_descriptor_save.close()
    descriptor_file_load.close()

    stacking_descriptor_load = h5py.File(_TEMP_DESC_STACK_FILE, "r")
    features = stacking_descriptor_load.get('stack_kmeans')

    # Perform k-means clustering
    print("kmeans...")
    kmeans_iterations = 1
    k_centroids = 100
    # centroid codebook is the vocabulary
    centroid_codebook, distortion = kmeans(features, k_centroids, kmeans_iterations)
    stacking_descriptor_load.close()

    # making the histogram with all the words (visual vocabulary)
    print("extracting histogram...")
    descriptor_file_load = h5py.File(_TEMP_DESC_FILE, "r")
    histogram_with_features = np.zeros((num_total_images, len(centroid_codebook)), 'float32')
    for row_id in range(num_total_images):
        desc = descriptor_file_load.get(str(row_id))
        words, dist = vq(desc, centroid_codebook)
        for w in words:
            histogram_with_features[row_id][w] += 1

    descriptor_file_load.close()  # close in the end
    # Scale histogram words
    print("scale histogram...")
    stdScaler = StandardScaler().fit(histogram_with_features)
    histogram_with_features = stdScaler.transform(histogram_with_features)

    # Linear SVM
    print("SVM...")
    clf = svm.LinearSVC()
    clf.fit(histogram_with_features, np.array(image_label_list))

    # Save the trained data
    print("Saving data in bow.pkl")
    joblib.dump((possible_labels_folders, centroid_codebook, clf, stdScaler), "bow.pkl", compress=3)

    remove(_TEMP_DESC_FILE)
    remove(_TEMP_DESC_STACK_FILE)


if __name__ == "__main__":
    train()
