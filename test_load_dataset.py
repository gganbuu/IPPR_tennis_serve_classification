from Load_dataset import load_datasets, load_test_datasets

# Define the base path for the dataset
base_path = 'datasets/serveDataset/'  # Set the base path here

# Function to count class occurrences
def count_classes(labels):
    class_counts = {0: 0, 1: 0}
    for lab in labels:
        class_counts[lab] += 1
    return class_counts

# Test loading training datasets
try:
    images, labels, keypoints = load_datasets(base_path)
    print("Training datasets loaded successfully!")
    print(f"Number of training images: {len(images)}")
    print(f"Number of training labels: {len(labels)}")
    print(f"Number of training keypoints: {len(keypoints)}")
    
    # Count and print the number of images for each class in the training dataset
    train_class_counts = count_classes(labels)
    print(f"Training Class 0 count: {train_class_counts[0]}")
    print(f"Training Class 1 count: {train_class_counts[1]}")
    
except Exception as e:
    print(f"Error loading training datasets: {e}")

# Test loading test datasets
try:
    test_images, test_labels, test_keypoints = load_test_datasets(base_path)
    print("Test datasets loaded successfully!")
    print(f"Number of test images: {len(test_images)}")
    print(f"Number of test labels: {len(test_labels)}")
    print(f"Number of test keypoints: {len(test_keypoints)}")
    
    # Count and print the number of images for each class in the test dataset
    test_class_counts = count_classes(test_labels)
    print(f"Test Class 0 count: {test_class_counts[0]}")
    print(f"Test Class 1 count: {test_class_counts[1]}")
    
except Exception as e:
    print(f"Error loading test datasets: {e}")
