import numpy as np
import os
import cv2
import random
from matplotlib import pyplot as plt
import pandas as pd
import logging
from typing import List
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
import datetime

DATASET_FOLDER = r'C:/Users/chiki/Downloads/notMNIST_large'
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
DATA_COLUMN_NAME = 'data'
LABELS_COLUMN_NAME = 'labels'
HASHED_DATA_COLUMN_NAME = 'data_bytes'
BALANCE_BORDER = 0.85
MAX_ITERATIONS_COUNT = 100
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


def is_class_balanced(class_images_count):
    max_images_count = max(class_images_count)
    avg_images_count = sum(class_images_count) / len(class_images_count)
    balance_percent = avg_images_count / max_images_count
    logging.info(f"Balance: {balance_percent:.3f}")

    return True if balance_percent > BALANCE_BORDER else False


def analyze_all_classes():
    """
    Analyze all classes:

    1 Check if classes are balanced

    2 Create data frame

    :return: data - list of ndarrays
    """
    images_counts_list = list()
    data = list()
    labels = list()
    for class_item in CLASSES:
        class_folder_path = os.path.join(DATASET_FOLDER, class_item)
        class_data = get_class_data(class_folder_path)

        class_images_count = len(class_data)
        logging.info(f"{class_item} folder contains {class_images_count} files")

        images_counts_list.append(class_images_count)
        data.extend(class_data)
        labels.extend([CLASSES.index(class_item) for _ in range(class_images_count)])

    if is_class_balanced(images_counts_list):
        logging.info("Classes are balanced")
    else:
        logging.info("Classes are not balanced")

    data_frame = pd.DataFrame({DATA_COLUMN_NAME: data, LABELS_COLUMN_NAME: labels})

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

    return x_train, y_train, x_test, y_test, x_valid, y_valid


def get_logistic_regression(x_train, y_train, x_test):
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=MAX_ITERATIONS_COUNT)
    logistic_regression.fit(x_train, y_train)
    logging.info("Regression fit is completed")

    logistic_regression.predict(x_test)
    logging.info("Predict is completed")

    train_sizes, train_scores, test_scores = learning_curve(
        logistic_regression,
        x_train,
        y_train,
        train_sizes=TRAIN_SIZES
    )

    return train_sizes, train_scores, test_scores


def show_result_plot(train_sizes, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title('Learning curve')
    plt.xlabel('Training data size')
    plt.ylabel('Accuracy')
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.25,
        color='r'
    )

    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Training score')
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.25,
        color='g'
    )

    plt.show()
    logging.info("Plot shown")


def main():
    start_time = datetime.datetime.now()

    show_some_images()

    data_frame = analyze_all_classes()
    data_frame = remove_duplicates(data_frame)
    data_frame = shuffle_data(data_frame)

    x_train, y_train, x_test, *_ = split_dataset_into_subsamples(data_frame)

    train_sizes, train_scores, test_scores = get_logistic_regression(x_train, y_train, x_test)
    show_result_plot(train_sizes, train_scores, test_scores)

    end_time = datetime.datetime.now()
    logging.info(end_time - start_time)


main()
