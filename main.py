import os
import cv2
import random
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

DATASET_FOLDER = r'C:/Users/chiki/Downloads/notMNIST_large'
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
DATA_COLUMN_NAME = 'data'
LABELS_COLUMN_NAME = 'labels'
HASHED_DATA_COLUMN_NAME = 'data_bytes'
BALANCE_BORDER = 0.85
MAX_ITERATIONS_COUNT = 200000
TRAIN_SIZES = [50, 100, 1000, 10000, 50000]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def show_one_image(image_folder):
    file = random.choice(os.listdir(image_folder))
    image_path = os.path.join(image_folder, file)
    img = cv2.imread(image_path)
    plt.imshow(img)
    plt.show()


def show_some_images():
    for class_item in CLASSES:
        image_folder = os.path.join(DATASET_FOLDER, class_item)
        show_one_image(image_folder)


def get_class_data(folder_path):
    result_data = list()
    files = os.listdir(folder_path)
    for file in files:
        image_path = os.path.join(folder_path, file)
        img = cv2.imread(image_path)
        if img is not None:
            result_data.append(img.reshape(-1))

    return result_data


def get_classes_images_counts(data_frame):
    classes_images_counts = list()
    for class_index in range(len(CLASSES)):
        labels = data_frame[LABELS_COLUMN_NAME]
        class_rows = data_frame[labels == class_index]
        class_count = len(class_rows)

        classes_images_counts.append(class_count)
        logging.info(f"Class {CLASSES[class_index]} contains {class_count} images")

    return classes_images_counts


def show_classes_histogram(classes_images_counts):
    plt.figure()
    plt.bar(CLASSES, classes_images_counts)
    plt.show()
    logging.info("Histogram shown")


def check_classes_balance(data_frame):
    classes_images_counts = get_classes_images_counts(data_frame)

    max_images_count = max(classes_images_counts)
    avg_images_count = sum(classes_images_counts) / len(classes_images_counts)
    balance_percent = avg_images_count / max_images_count

    show_classes_histogram(classes_images_counts)
    logging.info(f"Balance: {balance_percent:.3f}")
    if balance_percent > BALANCE_BORDER:
        logging.info("Classes are balanced")
    else:
        logging.info("Classes are not balanced")


def create_data_frame():
    data = list()
    labels = list()
    for class_item in CLASSES:
        class_folder_path = os.path.join(DATASET_FOLDER, class_item)
        class_data = get_class_data(class_folder_path)

        data.extend(class_data)
        labels.extend([CLASSES.index(class_item) for _ in range(len(class_data))])

    data_frame = pd.DataFrame({DATA_COLUMN_NAME: data, LABELS_COLUMN_NAME: labels})
    logging.info("Data frame is created")

    return data_frame


def remove_duplicates(data):
    data_bytes = [item.tobytes() for item in data[DATA_COLUMN_NAME]]
    data[HASHED_DATA_COLUMN_NAME] = data_bytes
    data.sort_values(HASHED_DATA_COLUMN_NAME, inplace=True)
    data.drop_duplicates(subset=HASHED_DATA_COLUMN_NAME, keep='first', inplace=True)
    data.pop(HASHED_DATA_COLUMN_NAME)
    logging.info("Duplicates removed")

    return data


def shuffle_data(data):
    data_shuffled = data.sample(frac=1, random_state=42)
    logging.info("Data shuffled")

    return data_shuffled


def split_dataset_into_subsamples(data_frame):
    data = np.array(list(data_frame[DATA_COLUMN_NAME].values), np.float32)
    labels = np.array(list(data_frame[LABELS_COLUMN_NAME].values), np.float32)

    x_train, x_remaining, y_train, y_remaining = train_test_split(data, labels, train_size=0.8)
    x_valid, x_test, y_valid, y_test = train_test_split(x_remaining, y_remaining, test_size=0.5)
    logging.info("Data split")

    return x_train, y_train, x_test, y_test, x_valid, y_valid


def get_logistic_regression(x_train, y_train, x_test, y_test):
    test_scores = list()
    for train_size in TRAIN_SIZES:
        logistic_regression = LogisticRegression(max_iter=MAX_ITERATIONS_COUNT)
        logistic_regression.fit(x_train[:train_size], y_train[:train_size])
        logging.info("Regression fit is completed")

        score = logistic_regression.score(x_test, y_test)
        logging.info("Score is calculated")
        test_scores.append(score)

    return test_scores


def show_result_plot(test_scores):
    plt.figure()
    plt.title('Learning curve')
    plt.xlabel('Training data size')
    plt.ylabel('Accuracy')
    plt.grid()

    plt.plot(TRAIN_SIZES, test_scores, 'o-', color='g', label='Testing score')

    plt.show()
    logging.info("Plot shown")


def main():
    start_time = datetime.datetime.now()

    show_some_images()

    data_frame = create_data_frame()
    data_frame = remove_duplicates(data_frame)
    check_classes_balance(data_frame)
    data_frame = shuffle_data(data_frame)

    x_train, y_train, x_test, y_test, *_ = split_dataset_into_subsamples(data_frame)

    test_scores = get_logistic_regression(x_train, y_train, x_test, y_test)
    show_result_plot(test_scores)

    end_time = datetime.datetime.now()
    logging.info(end_time - start_time)


main()
