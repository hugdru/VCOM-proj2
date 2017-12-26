from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import svm
from scipy.cluster.vq import vq, kmeans
from os import listdir, path
import numpy as np
import cv2

# Create sift for feature extraction
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

# path to aid folder
train_data_set_path = "../AID/"


def main():
    # get all the folders inside the aid folder(classes)
    possible_labels_folders = listdir(train_data_set_path)

    des_list = []
    num_total_images = 0
    image_label_list = []
    label = 0
    for label_folder in possible_labels_folders:
        label_folder_path = path.join(train_data_set_path, label_folder)

        # for all files in image folder(class)
        for image_file in listdir(label_folder_path):
            image_path = path.join(label_folder_path, image_file)
            print("Extracting descriptors (", label, "/", len(possible_labels_folders), "): ", image_path)
            # reading image in rgb
            im = cv2.imread(image_path)
            # extracting features
            kpts, des = surf.detectAndCompute(im, None)
            if des is not None:
                des_list.append(des)
                image_label_list += [label]
                num_total_images += 1
            else:
                print("Error extracting the descriptor: ", image_path)
        label += 1

    # using vstack to stack all the descriptors in one big numpy array to feed kmeans
    features = des_list[0]
    des_count = 1
    for des in des_list[1:]:
        print("Stacking descriptors for kmeans (", (des_count / num_total_images) * 100, "%)")
        des_count += 1
        features = np.vstack((features, des))

    # Perform k-means clustering
    print("kmeans...")
    kmeans_iterations = 1
    k_centroids = 100
    # centroid codebook is the vocabulary
    centroid_codebook, distortion = kmeans(features, k_centroids, kmeans_iterations)

    # making the histogram with all the words (visual vocabulary)
    print("extracting histogram...")
    histogram_with_features = np.zeros((num_total_images, len(centroid_codebook)), 'float32')
    for i in range(num_total_images):
        words, dist = vq(des_list[i], centroid_codebook)
        for w in words:
            histogram_with_features[i][w] += 1

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


if __name__ == "__main__":
    main()
