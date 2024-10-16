# test_load_dataset.py

from Load_dataset import load_datasets, load_test_datasets

# Define the base path for the dataset
base_path = 'datasets/tennis serve.v2i.coco(1)/'  # Set the base path here

# Test loading training datasets
try:
    images, labels, keypoints = load_datasets(base_path)
    print("Training datasets loaded successfully!")
    print(f"Number of training images: {len(images)}")
    print(f"Number of training labels: {len(labels)}")
    print(f"Number of training keypoints: {len(keypoints)}")
except Exception as e:
    print(f"Error loading training datasets: {e}")

# Test loading test datasets
try:
    test_images, test_labels, test_keypoints = load_test_datasets(base_path)
    print("Test datasets loaded successfully!")
    print(f"Number of test images: {len(test_images)}")
    print(f"Number of test labels: {len(test_labels)}")
    print(f"Number of test keypoints: {len(test_keypoints)}")
except Exception as e:
    print(f"Error loading test datasets: {e}")
