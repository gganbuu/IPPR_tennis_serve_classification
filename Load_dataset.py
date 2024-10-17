# load_dataset.py

import os
import numpy as np
import tensorflow as tf

# Load and preprocess images from a directory
def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Adjust for PNG if needed
    img = tf.image.resize(img, [224, 224])  # Resize to 224x224
    img = img / 255.0  # Normalise to [0, 1]
    return img

def load_images_from_directory(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust as needed
            img_path = os.path.join(directory, filename)
            img = load_and_preprocess_image(img_path)
            images.append(img)
            labels.append(label)
    return images, labels

# Load keypoints from a file
def load_keypoints_from_file(file_path):
    keypoints = []
    with open(file_path, 'r') as f:
        for line in f:
            coords = list(map(float, line.strip().split()))
            keypoints.append(coords)
    return np.array(keypoints)

# Normalise keypoints
def normalise_keypoints(keypoints, img_width, img_height):
    return keypoints / np.array([img_width, img_height])

# Load datasets
def load_datasets(base_path):
    class_0_images = []
    class_0_labels = []
    class_0_keypoints = []

    class_1_images = []
    class_1_labels = []
    class_1_keypoints = []

    for i in range(0, 3):
        directory = os.path.join(base_path, 'train/bad', f'train_bad_p{i}_pose_output')
        images_temp, labels_temp = load_images_from_directory(directory, 0)
        class_0_images.extend(images_temp)
        class_0_labels.extend(labels_temp)

        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                keypoints_path = os.path.join(directory, filename)
                keypoints = load_keypoints_from_file(keypoints_path)
                normalised_keypoints = normalise_keypoints(keypoints, 224, 224)  # Assuming 224x224 image size
                class_0_keypoints.append(normalised_keypoints)

    for i in range(1, 24):
        directory = os.path.join(base_path, 'train/good', f'train_good_p{i}_pose_output')
        images_temp, labels_temp = load_images_from_directory(directory, 0)
        class_1_images.extend(images_temp)
        class_1_labels.extend(labels_temp)

        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                keypoints_path = os.path.join(directory, filename)
                keypoints = load_keypoints_from_file(keypoints_path)
                normalised_keypoints = normalise_keypoints(keypoints, 224, 224)  # Assuming 224x224 image size
                class_1_keypoints.append(normalised_keypoints)


    # Combine the images, labels, and keypoints
    images = np.array(class_0_images + class_1_images)
    labels = np.array(class_0_labels + class_1_labels)
    keypoints = np.array(class_0_keypoints + class_1_keypoints)
    return images, labels, keypoints

# Load test datasets
def load_test_datasets(base_path):
    class_0_test_images = []
    class_0_test_labels = []
    class_0_test_keypoints = []

    class_1_test_images = []
    class_1_test_labels = []
    class_1_test_keypoints = []

    # Load images, labels, and keypoints for class 0 (p8 to p11)
    for i in range(8, 11):
        directory = os.path.join(base_path, 'test', f'p{i}_pose_output')
        images_temp, labels_temp = load_images_from_directory(directory, 0)
        class_0_test_images.extend(images_temp)
        class_0_test_labels.extend(labels_temp)

        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                keypoints_path = os.path.join(directory, filename)
                keypoints = load_keypoints_from_file(keypoints_path)
                normalised_keypoints = normalise_keypoints(keypoints, 224, 224)  # Assuming 224x224 image size
                class_0_test_keypoints.append(normalised_keypoints)

    # Load images, labels, and keypoints for class 1 (p1 to p7)
    for i in range(1, 8):
        directory = os.path.join(base_path, 'test', f'p{i}_pose_output')
        images_temp, labels_temp = load_images_from_directory(directory, 1)
        class_1_test_images.extend(images_temp)
        class_1_test_labels.extend(labels_temp)

        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                keypoints_path = os.path.join(directory, filename)
                keypoints = load_keypoints_from_file(keypoints_path)
                normalised_keypoints = normalise_keypoints(keypoints, 224, 224)
                class_1_test_keypoints.append(normalised_keypoints)

    # Combine the test images, labels, and keypoints for both classes
    test_images = np.array(class_0_test_images + class_1_test_images)
    test_labels = np.array(class_0_test_labels + class_1_test_labels)
    test_keypoints = np.array(class_0_test_keypoints + class_1_test_keypoints)

    return test_images, test_labels, test_keypoints
