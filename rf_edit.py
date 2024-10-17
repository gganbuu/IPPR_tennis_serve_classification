import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Base path to your dataset
base_path = 'datasets/tennis serve.v2i.coco(1)/'

# Function to flatten and combine features
def prepare_features(images, keypoints):
    flattened_images = [img.numpy().flatten() for img in images]
    flattened_images = np.array(flattened_images)
    flattened_keypoints = keypoints.reshape(keypoints.shape[0], -1)
    features = np.hstack((flattened_images, flattened_keypoints))
    return features

# Load images and keypoints from a directory
def load_images_and_keypoints(directory, label):
    images = []
    labels = []
    keypoints = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(directory, filename)
            img = load_and_preprocess_image(img_path)
            images.append(img)
            labels.append(label)
        elif filename.endswith('.txt'):
            keypoints_path = os.path.join(directory, filename)
            kp = load_keypoints_from_file(keypoints_path)
            normalised_keypoints = normalise_keypoints(kp, 224, 224)
            keypoints.append(normalised_keypoints)
    return images, labels, keypoints

# Load datasets
def load_datasets(base_path):
    # Load 'bad' serves (class 0)
    bad_directory = os.path.join(base_path, 'Baddataset')
    bad_images, bad_labels, bad_keypoints = [], [], []
    for i in range(0, 5):  # Adjust if you have different numbers
        directory = os.path.join(bad_directory, f'WA0{i}_pose_output')
        images, labels, keypoints = load_images_and_keypoints(directory, 0)
        bad_images.extend(images)
        bad_labels.extend(labels)
        bad_keypoints.extend(keypoints)

    # Load 'good' serves (class 1)
    good_train_directory = os.path.join(base_path, 'train_pose_output')
    good_valid_directory = os.path.join(base_path, 'valid_pose_output')
    good_images_train, good_labels_train, good_keypoints_train = load_images_and_keypoints(good_train_directory, 1)
    good_images_valid, good_labels_valid, good_keypoints_valid = load_images_and_keypoints(good_valid_directory, 1)

    # Combine all images, labels, and keypoints for training
    images = np.array(bad_images + good_images_train + good_images_valid)
    labels = np.array(bad_labels + good_labels_train + good_labels_valid)
    keypoints = np.array(bad_keypoints + good_keypoints_train + good_keypoints_valid)

    return images, labels, keypoints

# Load test datasets
def load_test_datasets(base_path):
    test_directory = os.path.join(base_path, 'test_pose_output')
    test_images, test_labels, test_keypoints = load_images_and_keypoints(test_directory, 1)  # Label 1 for 'good'
    # You might want to add logic for 'bad' test cases if available in the structure

    return np.array(test_images), np.array(test_labels), np.array(test_keypoints)

# Load the training dataset
images, labels, keypoints = load_datasets(base_path)

# Prepare features for training
X_train = prepare_features(images, keypoints)
y_train = labels

# Load the test dataset
test_images, test_labels, test_keypoints = load_test_datasets(base_path)

# Prepare features for testing
X_test = prepare_features(test_images, test_keypoints)
y_test = test_labels

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
