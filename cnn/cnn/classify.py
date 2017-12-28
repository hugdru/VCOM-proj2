import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.preprocessing.csv_generator import CSVGenerator

import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os

import tensorflow as tf

# Adapted from https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet%20-%20COCO%202017.ipynb

ANNOTATIONS = "csv_data/testing_annotations.csv"
CLASSES = "csv_data/classes.csv"
BATCH_SIZE = 1
BASE_DIR = "./"
SNAPSHOT_DIR = "snapshots/"
# SNAPSHOT_FILE = SNAPSHOT_DIR + "resnet50_csv_50.h5"
SNAPSHOT_FILE = None


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    keras.backend.tensorflow_backend.set_session(get_session())

    if SNAPSHOT_FILE:
        snapshot_file = SNAPSHOT_FILE
    else:
        snapshots = os.listdir(SNAPSHOT_DIR)
        if not snapshots:
            raise Exception("There are no snapshots")

        snapshot_file = os.path.join(SNAPSHOT_DIR,
                                     sorted(snapshots, reverse=True)[0])

    print("snapshot: {}".format(snapshot_file))

    model = keras.models.load_model(
        snapshot_file, custom_objects=custom_objects)
    # print(model.summary())

    # create image data generator object
    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    # create a generator for testing data
    val_generator = CSVGenerator(
        ANNOTATIONS,
        CLASSES,
        val_image_data_generator,
        batch_size=BATCH_SIZE,
        base_dir=BASE_DIR)

    index = 0

    # load image
    image = val_generator.load_image(index)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = val_generator.preprocess_image(image)
    image, scale = val_generator.resize_image(image)
    annotations = val_generator.load_annotations(index)
    index += 1

    # process image
    start = time.time()
    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # compute predicted labels and scores
    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0,
                        np.arange(detections.shape[1]), 4 + predicted_labels]

    # correct for image scale
    detections[0, :, :4] /= scale

    # visualize detections
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score < 0.5:
            continue
        b = detections[0, idx, :4].astype(int)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
        caption = "{} {:.3f}".format(val_generator.label_to_name(label), score)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 0, 0), 3)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (255, 255, 255), 2)

    # visualize annotations
    for annotation in annotations:
        label = int(annotation[4])
        b = annotation[:4].astype(int)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        caption = "{}".format(val_generator.label_to_name(label))
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 0, 0), 3)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (255, 255, 255), 2)

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()


if __name__ == "__main__":
    main()
