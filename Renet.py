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

        # Apply augmentations for certain labels (1 or 0 depending on the 'check' flag)
        if check:
            if lab == 1:
                for _ in range(3):  # Apply transformations 3 times for label 1
                    transformed_img, transformed_kp = apply_transformations(original_img_tensor.clone(), original_kp_tensor.clone())
                    image_tensors.append(transformed_img)
                    keypoint_tensors.append(transformed_kp)
                    label_list.append(lab)
        else:
            if lab == 0:
                for _ in range(1):  # Apply transformations 1 time for label 0
                    transformed_img, transformed_kp = apply_transformations(original_img_tensor.clone(), original_kp_tensor.clone())
                    image_tensors.append(transformed_img)
                    keypoint_tensors.append(transformed_kp)
                    label_list.append(lab)

    images_tensor = torch.stack(image_tensors)
    keypoints_tensor = torch.stack(keypoint_tensors)
    labels_tensor = torch.tensor(label_list, dtype=torch.float32)
    
    # Debug: Print shapes after transformations
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
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the input layer to accept 3-channel images (if necessary)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Identity()  # Remove final fully connected layer

        # Custom fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Add BatchNorm layer
        self.fc2 = nn.Linear(256 + 14, 1)  # Combine ResNet output with keypoints
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)
        
        # Initialize weights for custom layers
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, keypoints):
        x = self.resnet(x)
        keypoints_flat = keypoints.view(keypoints.size(0), -1)

        x = F.gelu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = torch.cat((x, keypoints_flat), dim=1)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# Instantiate the model, loss function, and optimizer
model = ServeResNet()
class_weights = torch.tensor([1.0, (191 / 200)]).float()  # Adjust weights if needed
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

# Save results to JSON
def save_results_to_json(test_labels, test_predictions, training_stats, output_path="classification_results_ResNet_leakyRELU.json"):
    class_report = classification_report(test_labels, test_predictions, output_dict=True, zero_division=1)
    conf_matrix = confusion_matrix(test_labels, test_predictions).tolist()

    results = {
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "training_stats": training_stats
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

# Train the model
def train(model, loader, optimizer, criterion, n_epochs=1, patience=5):
    best_loss = float('inf')
    best_model = None
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
                best_model = model.state_dict()  # Save the best model state

    # Load the best model state
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, training_stats

# Evaluate the model on the test dataset
def evaluate(model, loader):
    model.eval()
    total_correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels, keypoints in loader:
            outputs = model(images, keypoints).squeeze(1)
            preds = (outputs >= 0.05).float()  # Apply threshold for binary predictions
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.tolist())
            all_predictions.extend(preds.tolist())

    accuracy = total_correct / total
    return all_labels, all_predictions, accuracy

# Train the model
trained_model, training_stats = train(model, train_loader, optimizer, criterion, n_epochs=10)

# Evaluate the model
test_labels, test_predictions, test_accuracy = evaluate(trained_model, test_loader)

# Save the classification report and confusion matrix
save_results_to_json(test_labels, test_predictions, training_stats)
