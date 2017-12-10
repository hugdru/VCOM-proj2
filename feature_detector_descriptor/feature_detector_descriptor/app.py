from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2 as cv


def main():
    image = cv.imread("../AID/Airport/airport_1.jpg")

    hist = cv.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                       [0, 256, 0, 256, 0, 256])

    print(hist)


if __name__ == "__main__":
    main()
