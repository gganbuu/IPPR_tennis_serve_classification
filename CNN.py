import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import json  # For saving the classification report and confusion matrix

# Load datasets (assuming these functions are defined in Load_dataset.py)
from Load_dataset import load_datasets, load_test_datasets

# Define the base path for the dataset
base_path = 'datasets/serveDataset/'  # Set the correct path

# Load datasets
images, labels, keypoints = load_datasets(base_path)
test_images, test_labels, test_keypoints = load_test_datasets(base_path)

print(f"test Images shape: {np.array(test_images).shape}")
print(f"test Labels shape: {np.array(test_labels).shape}")
print(f"Original test Keypoints shape: {np.array(test_keypoints).shape}")

# Count classes
def count_classes(labels):
    class_counts = {0: 0, 1: 0}
    for lab in labels:
        class_counts[lab] += 1
    return class_counts

# Count and print the number of images for each class in the original test dataset
original_test_class_counts = count_classes(test_labels)
print(f"Original Test Class 0 count: {original_test_class_counts[0]}")
print(f"Original Test Class 1 count: {original_test_class_counts[1]}")
train_class_counts = count_classes(labels)
print(f"Original train Class 0 count: {train_class_counts[0]}")
print(f"Original train Class 1 count: {train_class_counts[1]}")

# Define the rotate_image_and_keypoints and apply_transformations functions
def rotate_image_and_keypoints(image, keypoints, angle):
    rotated_image = TF.rotate(image, angle)
    theta = math.radians(angle)
    Cx, Cy = image.shape[2] / 2, image.shape[1] / 2
    rotation_matrix = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])
    shifted_keypoints = keypoints - torch.tensor([Cx, Cy])
    rotated_keypoints = (rotation_matrix @ shifted_keypoints.T).T
    rotated_keypoints += torch.tensor([Cx, Cy])
    return rotated_image, rotated_keypoints

def apply_transformations(image, keypoints):
    angle = torch.FloatTensor(1).uniform_(-30, 30).item()
    image, keypoints = rotate_image_and_keypoints(image, keypoints, angle)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image), keypoints

# Prepare the datasets with transformations
def prepare_dataset(images, labels, keypoints, check):
    image_tensors, keypoint_tensors = [], []
    label_list = []

    for img, kp, lab in zip(images, keypoints, labels):
        original_img_tensor = torch.tensor(img).permute(2, 0, 1)
        original_kp_tensor = torch.tensor(kp, dtype=torch.float32)
        image_tensors.append(original_img_tensor)
        keypoint_tensors.append(original_kp_tensor)
        label_list.append(lab)

        # If the label is 1, also apply transformations and store them
        if check:
            if lab == 1:
                for _ in range(3):  # Apply transformations 3 times
                    transformed_img, transformed_kp = apply_transformations(original_img_tensor.clone(), original_kp_tensor.clone())
                    image_tensors.append(transformed_img)
                    keypoint_tensors.append(transformed_kp)
                    label_list.append(lab)
        else:
            if lab == 0:
                for _ in range(1):  # Apply transformations 2 times
                    transformed_img, transformed_kp = apply_transformations(original_img_tensor.clone(), original_kp_tensor.clone())
                    image_tensors.append(transformed_img)
                    keypoint_tensors.append(transformed_kp)
                    label_list.append(lab)

    images_tensor = torch.stack(image_tensors)
    keypoints_tensor = torch.stack(keypoint_tensors)
    labels_tensor = torch.tensor(label_list, dtype=torch.long)  # CrossEntropyLoss expects integer labels
    test_class_count = count_classes(label_list)
    print(f"New Test Class 0 count: {test_class_count[0]}")
    print(f"New Test Class 1 count: {test_class_count[1]}")
    # Print shapes after transformations
    print(f"Images shape after augmentations: {images_tensor.shape}")
    print(f"Labels shape after augmentations: {labels_tensor.shape}")
    print(f"Keypoints shape after augmentations: {keypoints_tensor.shape}")

    return images_tensor, labels_tensor, keypoints_tensor

# Prepare the dataset with transformations applied
train_images_tensor, train_labels_tensor, train_keypoints_tensor = prepare_dataset(images, labels, keypoints, check=False)
test_images_tensor, test_labels_tensor, test_keypoints_tensor = prepare_dataset(test_images, test_labels, test_keypoints, check=True)

# Create DataLoaders
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor, train_keypoints_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(test_images_tensor, test_labels_tensor, test_keypoints_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the CNN model
class ServeCNN(nn.Module):
    def __init__(self):
        super(ServeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Single conv layer
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization for conv1
        self.fc1 = nn.Linear(16 * 112 * 112 + 14, 64)  # Adjusted for concatenation with 14 keypoint dimensions
        self.fc2 = nn.Linear(64, 2)  # Output layer
        self.dropout = nn.Dropout(p=0.5)  # Reduced dropout rate

    def forward(self, x, keypoints):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Convolution + ReLU + Pooling
        x = x.view(-1, 16 * 112 * 112)  # Flattening the tensor; this will be (N, 200704)

        # Flatten the keypoints and concatenate with the flattened features
        keypoints_flat = keypoints.view(keypoints.size(0), -1)  # Flatten keypoints to (N, 14)
        x = torch.cat((x, keypoints_flat), dim=1)  # Concatenate along the feature dimension
        
        x = self.dropout(F.relu(self.fc1(x)))  # Fully connected layer with ReLU
        x = self.fc2(x)  
        return x  
# Instantiate the model, loss function, and optimizer
model = ServeCNN()
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

# Generate classification report and confusion matrix, then save them to a JSON file
def save_results_to_json(test_labels, test_predictions, train_accuracy, train_loss, test_accuracy, test_loss, output_path="CrossEntropyLoss_results_SGD_lr=0.0001.json"):
    # Convert test labels and predictions to NumPy arrays
    test_labels_np = test_labels.numpy()
    test_predictions_np = np.array(test_predictions)

    # Classification report and confusion matrix
    class_report = classification_report(test_labels_np, test_predictions_np, output_dict=True, zero_division=1)
    conf_matrix = confusion_matrix(test_labels_np, test_predictions_np).tolist()

    # Create a dictionary to store the results
    results = {
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "training_accuracy": train_accuracy,
        "training_loss": train_loss,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "training_stats": training_stats
    }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

def train(model, loader, optimizer, criterion, epochs=1):
    model.train()
    training_stats = {
        "epochs": [],
        "losses": [],
        "accuracies": []
    }
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels, keypoints in tqdm(loader):
            optimizer.zero_grad()
            outputs = model(images, keypoints)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)  # Argmax for classification
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(loader)
        avg_accuracy = correct / total
        
        # Append the stats for the epoch
        training_stats["epochs"].append(epoch + 1)
        training_stats["losses"].append(avg_loss)
        training_stats["accuracies"].append(avg_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return avg_accuracy, avg_loss, training_stats


def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions_list = []

    with torch.no_grad():
        for images, labels, keypoints in loader:
            outputs = model(images, keypoints)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            predictions_list.extend(predictions.cpu().numpy())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    avg_accuracy = correct / total
    return avg_accuracy, avg_loss, predictions_list

# Training and testing the model
train_accuracy, train_loss, training_stats = train(model, train_loader, optimizer, criterion, epochs=5)
test_accuracy, test_loss, test_predictions = test(model, test_loader, criterion)

# Save results to JSON
save_results_to_json(test_labels_tensor, test_predictions, train_accuracy, train_loss, test_accuracy, test_loss)
