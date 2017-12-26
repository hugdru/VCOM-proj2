from os import path, makedirs
from shutil import move
from glob import glob
import math
import random

PATH = "../AID/"
percentage_to_test = 0.4


def main():
    for each in glob(PATH + "*"):
        classs = each.split("/")[-1]
        print("Reading class folder: ", classs)

        all_word_imgs = []
        n = 0
        for img_f in glob(PATH + classs + "/*.jpg"):
            print("detecting file: ", img_f)
            all_word_imgs.append(img_f)
            n += 1

        # separate images
        random.shuffle(all_word_imgs)
        train_size = math.floor((1.0 - percentage_to_test) * n)

        train_imgs = all_word_imgs[:train_size]
        test_imgs = all_word_imgs[train_size:]

        test_file_path = '../AID_test' + '/' + classs + '/'
        makedirs(path.dirname(test_file_path))
        for img_f in test_imgs:
            print("save file: ", test_file_path + img_f.split("/")[-1])
            move(img_f, test_file_path + img_f.split("/")[-1])


if __name__ == "__main__":
    main()
