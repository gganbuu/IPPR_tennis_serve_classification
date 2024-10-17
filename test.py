
import os
import sys

# Add the path to the directory containing Load_dataset.py


# Now import the function
from Load_dataset import load_test_datasets
from Load_dataset import load_datasets

# Define your base path
relative_path = r"datasets\tennis serve.v2i.coco(1)"
BASE_PATH = os.path.join(os.getcwd(), relative_path)
# Load the test dataset
try:
    test_images, test_labels, test_keypoints = load_test_datasets(BASE_PATH)
    images,labels,keypoints = load_datasets(BASE_PATH)
    print(f"Test Images shape: {test_images.shape}")
    print(f"Test Labels shape: {test_labels.shape}")
    print(f"Test Keypoints shape: {test_keypoints.shape}")
    print(f" Images shape: {images.shape}")
    print(f" Labels shape: {labels.shape}")
    print(f" Keypoints shape: {keypoints.shape}")

except Exception as e:
    print(f"An error occurred: {e}")
