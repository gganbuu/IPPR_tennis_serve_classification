import os
import cv2
import numpy as np

from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def extract_sift_features(data_dir):
    features = []
    labels = []
    
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()
    
    # Loop over each folder (person's serve) inside the directory
    for subdir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                # Load image using OpenCV
                img_path = os.path.join(subdir, file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
                
                # Resize image to a fixed size (e.g., 128x128) to ensure uniformity
                image = cv2.resize(image, (128, 128))

                # Detect keypoints and compute descriptors using SIFT
                keypoints, descriptors = sift.detectAndCompute(image, None)
                
                # If no descriptors are found, skip the image
                if descriptors is None:
                    continue
                
                # We use the sum of descriptors as a feature vector
                sift_features = np.sum(descriptors, axis=0)
                
                # Add the features and corresponding label (good or bad)
                features.append(sift_features)
                
                # Determine label from the filename (1 = good, 0 = bad)
                if "good" in file:
                    labels.append(1)
                else:
                    labels.append(0)

    # Convert to numpy arrays
    return np.array(features), np.array(labels)

# Paths to your datasets
train_dir = "/Users/rajkulkarni/Documents/UTS 2024/SPRING/Image Recognition/Project Code/IPPR_tennis_serve_classification/datasets/serveDataset/train"
test_dir = "/Users/rajkulkarni/Documents/UTS 2024/SPRING/Image Recognition/Project Code/IPPR_tennis_serve_classification/datasets/serveDataset/test"
valid_dir = "/Users/rajkulkarni/Documents/UTS 2024/SPRING/Image Recognition/Project Code/IPPR_tennis_serve_classification/datasets/serveDataset/valid"

# Extract SIFT features for each set
X_train, y_train = extract_sift_features(train_dir)
X_test, y_test = extract_sift_features(test_dir)
X_valid, y_valid = extract_sift_features(valid_dir)

# Train SVM on training data

def SVM_model(train_X, train_Y):
    model = svm.SVC(kernel='linear', class_weight='balanced')
    model.fit(X_train, y_train)
    
    return model
model = SVM_model(X_train, y_train)

# Evaluate on validation data
def SVM_validate(X_valid, y_valid):
    y_pred = model.predict(X_valid)
    f1_val = f1_score(y_valid, y_pred)
    print(f"Validation Accuracy: {accuracy_score(y_valid, y_pred)}")
    print(f"Validation F1 Score: {f1_val}")
    return [f1_val, accuracy_score(y_valid, y_pred)]

# Test the trained model
def SVM_test(X_test, y_test):
    y_test_pred = model.predict(X_test)
    f1_test = f1_score(y_test, y_test_pred)
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")
    print(f"Test F1 Score: {f1_test}")
    return [f1_test, accuracy_score(y_test, y_test_pred)]


validation_scores = SVM_validate(X_valid, y_valid)
testing_scores = SVM_test(X_test, y_test)

# cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
# disp.plot(cmap=plt.cm.Blues)
# plt.title("Confusion Matrix")
# plt.show()