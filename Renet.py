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
from torchvision import models  # Import models from torchvision

from Load_dataset import load_datasets, load_test_datasets  # Assuming these functions are defined

# Define the base path for the dataset
base_path = 'datasets/serveDataset/'

# Load datasets
images, labels, keypoints = load_datasets(base_path)
test_images, test_labels, test_keypoints = load_test_datasets(base_path)

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
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
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
                for _ in range(1):  # Apply transformations 1 time
                    transformed_img, transformed_kp = apply_transformations(original_img_tensor.clone(), original_kp_tensor.clone())
                    image_tensors.append(transformed_img)
                    keypoint_tensors.append(transformed_kp)
                    label_list.append(lab)

    images_tensor = torch.stack(image_tensors)
    keypoints_tensor = torch.stack(keypoint_tensors)
    labels_tensor = torch.tensor(label_list, dtype=torch.float32)
    
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

# Define the CNN model using ResNet
class ServeResNet(nn.Module):
    def __init__(self):
        super(ServeResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # Load pre-trained ResNet18
        # Modify the input layer to accept 3-channel images (if necessary)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Modify the final fully connected layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 64)  # Change output features to 64
        self.fc2 = nn.Linear(64 + 14, 1)  # Output layer for combined features (64 + 14 for keypoints)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer

    def forward(self, x, keypoints):
        x = self.resnet(x)  # Forward through ResNet
        keypoints_flat = keypoints.view(keypoints.size(0), -1)  # Flatten keypoints to (N, 14)
        x = torch.cat((x, keypoints_flat), dim=1)  # Concatenate ResNet features with keypoints
        x = self.dropout(F.leaky_relu(self.fc2(x), negative_slope=0.01))  # Fully connected layer with dropout
        return x  # No sigmoid applied here

# Instantiate the model, loss function, and optimizer
model = ServeResNet()
class_weights = torch.tensor([1.0, (191 / 200)]).float()
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

# Generate classification report and confusion matrix, then save them to a JSON file
def save_results_to_json(test_labels, test_predictions, training_stats, output_path="classification_results_Renet.json"):
    class_report = classification_report(test_labels, test_predictions, output_dict=True, zero_division=1)
    conf_matrix = confusion_matrix(test_labels, test_predictions).tolist()

    results = {
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "training_stats": training_stats
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

def train(model, loader, optimizer, criterion, n_epochs=1, patience=5):
    best_accuracy = 0.0
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None
    losses_bits = []  # Track losses
    training_stats = {
        "epochs": [],
        "losses": [],
        "accuracies": []
    }

    model.train()
    with tqdm(total=n_epochs, unit="epoch") as pbar:
        for epoch in range(n_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in loader:
                images, labels, keypoints = batch

                # Forward pass
                outputs = model(images, keypoints).squeeze(1)
                loss = criterion(outputs, labels.float())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss and accuracy
                total_loss += loss.item()
                preds = (outputs >= 0.05).float()  # Convert sigmoid output to binary predictions
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = total_loss / len(loader)
            epoch_accuracy = correct / total
            losses_bits.append(epoch_loss)

            # Store training stats
            training_stats["epochs"].append(epoch + 1)
            training_stats["losses"].append(epoch_loss)
            training_stats["accuracies"].append(epoch_accuracy)

            # Update tqdm description
            pbar.set_description(f"Epoch {epoch + 1} - Accuracy: {epoch_accuracy:.4f} - Loss: {epoch_loss:.4f}")
            pbar.update(1)

            # Check for improvements
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
                best_model = model.state_dict()  # Save the model state
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

    # Load best model state
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, losses_bits, training_stats

# Train the model
model, losses_bits, training_stats = train(model, train_loader, optimizer, criterion, n_epochs=2)

# Evaluate the model on the test dataset
def evaluate(model, loader):
    model.eval()
    total_correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images, labels, keypoints = batch
            outputs = model(images, keypoints).squeeze(1)
            preds = (outputs >= 0.05).float()  # Convert sigmoid output to binary predictions
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / total
    return accuracy, all_labels, all_predictions

# Get accuracy and predictions on the test set
accuracy, test_labels, test_predictions = evaluate(model, test_loader)

# Save results to JSON
save_results_to_json(test_labels, test_predictions, training_stats)

# Print the final accuracy
print(f"Test Accuracy: {accuracy:.4f}")
