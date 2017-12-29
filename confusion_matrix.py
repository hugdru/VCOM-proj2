import csv
import os
import argparse

CONFUSION_MATRIX_FILENAME = "confusion_matrix.csv"
CLASSES_FILEPATH = "AID_DIVISION/labels.csv"


def main():

    parser = argparse.ArgumentParser(
        description='Confusion matrix creator',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        '-cr',
        '--comparison-results',
        metavar='file',
        required=True,
        type=str,
        dest='comparison_results_filepath',
        help='Comparison results filepath')

    parser.add_argument(
        '-od',
        '--output-dir',
        metavar='dir',
        required=True,
        type=str,
        dest='output_dir',
        help='Output dir where to place the confusion_matrix.csv')

    args = parser.parse_args()

    classes_index = 0
    classes_dic = {}
    classes_with_insertion_order = []
    for (category, ) in read_csv(CLASSES_FILEPATH):
        classes_dic[category] = classes_index
        classes_with_insertion_order = classes_with_insertion_order + [
            category
        ]
        classes_index += 1

    confusion_matrix = build_confusion_matrix(classes_index)

    with open(args.comparison_results_filepath, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for (i, row) in enumerate(csv_reader):
            if i == 0:
                continue
            expected = row[-1]
            predicted = row[-2]
            confusion_matrix[classes_dic[expected]][classes_dic[
                predicted]] += 1

    confusion_matrix_filepath = os.path.join(args.output_dir,
                                             CONFUSION_MATRIX_FILENAME)

    with open(confusion_matrix_filepath, "w") as confusion_matrix_file:
        confusion_matrix_writer = csv.writer(confusion_matrix_file)
        confusion_matrix_writer.writerow([""] + classes_with_insertion_order)
        for i, class_ in enumerate(classes_with_insertion_order):
            confusion_matrix_writer.writerow([class_] + confusion_matrix[i])


def read_csv(csv_filepath):
    with open(csv_filepath, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            yield row


def build_confusion_matrix(n):
    return [[0 for x in range(n)] for y in range(n)]


if __name__ == "__main__":
    main()
