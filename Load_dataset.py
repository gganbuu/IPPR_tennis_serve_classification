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

    for i in range(0, 2):  # Assuming you have WA00 to WA04
        # Constructing the path for class 0 images
        directory = os.path.join(base_path, 'Baddataset', f'WA0{i}_pose_output')  # Using the new base path
        class_0_images_temp, class_0_labels_temp = load_images_from_directory(directory, 0)
        class_0_images.extend(class_0_images_temp)
        class_0_labels.extend(class_0_labels_temp)
        
        # Load keypoints for each image in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                keypoints = load_keypoints_from_file(os.path.join(directory, filename))
                # Normalise keypoints
                normalised_keypoints = normalise_keypoints(keypoints, 224, 224)  # Assuming image size is 224x224
                class_0_keypoints.append(normalised_keypoints)

    # Load images and keypoints for class 1
    class_1_directory = os.path.join(base_path, 'train_pose_output')  # Using the new base path
    class_1_images, class_1_labels = load_images_from_directory(class_1_directory, 1)
    class_1_keypoints = []

    for filename in os.listdir(class_1_directory):
        if filename.endswith('.txt'):
            keypoints = load_keypoints_from_file(os.path.join(class_1_directory, filename))
            # Normalise keypoints
            normalised_keypoints = normalise_keypoints(keypoints, 224, 224)  # Assuming image size is 224x224
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

    for i in range(2, 5):  # Load WA03 and WA04 for class 0
        # Constructing the path for class 0 test images
        directory = os.path.join(base_path, 'Baddataset', f'WA0{i}_pose_output')  # Using the new base path
        class_0_test_images_temp, class_0_test_labels_temp = load_images_from_directory(directory, 0)
        class_0_test_images.extend(class_0_test_images_temp)
        class_0_test_labels.extend(class_0_test_labels_temp)

        # Load keypoints for each test image
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                keypoints_path = os.path.join(directory, filename)
                keypoints = load_keypoints_from_file(keypoints_path)
                # Normalise keypoints
                normalised_keypoints = normalise_keypoints(keypoints, 224, 224)  # Assuming image size is 224x224
                class_0_test_keypoints.append(normalised_keypoints)

    # Update this path to point to the correct directory for class 1 test images
    class_1_test_directory = os.path.join(base_path, 'test_pose_output')  # Using the new base path
    class_1_test_images, class_1_test_labels = load_images_from_directory(class_1_test_directory, 1)
    class_1_test_keypoints = []

    for filename in os.listdir(class_1_test_directory):
        if filename.endswith('.txt'):
            keypoints_path = os.path.join(class_1_test_directory, filename)
            keypoints = load_keypoints_from_file(keypoints_path)
            # Normalise keypoints
            normalised_keypoints = normalise_keypoints(keypoints, 224, 224)  # Assuming image size is 224x224
            class_1_test_keypoints.append(normalised_keypoints)

    # Combine the test images and labels
    test_images = np.array(class_0_test_images + class_1_test_images)
    test_labels = np.array(class_0_test_labels + class_1_test_labels)
    test_keypoints = np.array(class_0_test_keypoints + class_1_test_keypoints)
    return test_images, test_labels, test_keypoints
