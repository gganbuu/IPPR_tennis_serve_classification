import os
import cv2
import numpy as np

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
                print(img_path)
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

extract_sift_features("/Users/rajkulkarni/Documents/UTS 2024/SPRING/Image Recognition/Project Code/IPPR_tennis_serve_classification/datasets/serveDataset/train")
# print(extract_sift_features("/Users/rajkulkarni/Documents/UTS 2024/SPRING/Image Recognition/Project Code/IPPR_tennis_serve_classification/datasets/serveDataset/train"))