import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import classification_report

from Load_dataset import load_datasets, load_test_datasets  # Assuming these functions are defined

# Define the base path for the dataset
base_path = 'datasets/serveDataset/'

# Load datasets
images, labels, keypoints = load_datasets(base_path)
test_images, test_labels, test_keypoints = load_test_datasets(base_path)

# Rotate image and keypoints
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

# Random color jitter
def random_color_jitter():
    brightness = torch.FloatTensor(1).uniform_(0.1, 0.6).item()
    contrast = torch.FloatTensor(1).uniform_(0.1, 0.6).item()
    saturation = torch.FloatTensor(1).uniform_(0.1, 0.6).item()
    hue = torch.FloatTensor(1).uniform_(0, 0.2).item()
    return transforms.ColorJitter(brightness, contrast, saturation, hue)

# Apply transformations
def apply_transformations(image, keypoints):
    angle = torch.FloatTensor(1).uniform_(-30, 30).item()
    image, keypoints = rotate_image_and_keypoints(image, keypoints, angle)

    transform = transforms.Compose([
        random_color_jitter(),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image), keypoints

# Prepare dataset with transformations
def prepare_dataset(images, labels, keypoints):
    image_tensors, keypoint_tensors, label_list = [], [], []

    for img, kp, lab in zip(images, keypoints, labels):
        img_tensor, kp_tensor = apply_transformations(
            torch.tensor(img).permute(2, 0, 1), 
            torch.tensor(kp, dtype=torch.float32)
        )
        image_tensors.append(img_tensor)
        keypoint_tensors.append(kp_tensor)
        label_list.append(lab)

        # Duplicate data if label is 1
        if lab == 1:
            image_tensors.append(img_tensor.clone())
            keypoint_tensors.append(kp_tensor.clone())
            label_list.append(lab)

    images_tensor = torch.stack(image_tensors)
    keypoints_tensor = torch.stack(keypoint_tensors)
    labels_tensor = torch.tensor(label_list, dtype=torch.float32)

    return images_tensor, labels_tensor, keypoints_tensor

train_images_tensor, train_labels_tensor, train_keypoints_tensor = prepare_dataset(images, labels, keypoints)
test_images_tensor, test_labels_tensor, test_keypoints_tensor = prepare_dataset(test_images, test_labels, test_keypoints)

# Create DataLoaders
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor, train_keypoints_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(test_images_tensor, test_labels_tensor, test_keypoints_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the modified ResNet model
class ServeResNet(nn.Module):
    def __init__(self):
        super(ServeResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # Use ResNet-18
        self.resnet.fc = nn.Identity()  # Remove the default fully connected layer
        
        # Define new fully connected layers
        self.fc1 = nn.Linear(512 + 14, 64)  # 512 from ResNet + 14 keypoint dims
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization layer
        self.fc2 = nn.Linear(64, 1)  # Binary classification output
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, keypoints):
        x = self.resnet(x)  # Extract features with ResNet
        keypoints_flat = keypoints.view(keypoints.size(0), -1)  # Flatten keypoints
        x = torch.cat((x, keypoints_flat), dim=1)  # Concatenate features and keypoints
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.bn1(x)  # Apply batch normalization
        x = self.dropout(x)  # Dropout after batch normalization
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

# Instantiate the model, loss function, and optimizer
model = ServeResNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Training function
def train(model, loader, optimizer, criterion, n_epochs=1, patience=5):
    best_accuracy, best_loss = 0.0, float('inf')
    epochs_without_improvement = 0
    best_model = None

    model.train()
    with tqdm(total=n_epochs, unit="epoch") as pbar:
        for epoch in range(n_epochs):
            total_loss, correct, total = 0.0, 0, 0

            for images, labels, keypoints in loader:
                outputs = model(images, keypoints).squeeze(1)
                loss = criterion(outputs, labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = (outputs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = total_loss / len(loader)
            epoch_accuracy = correct / total

            pbar.set_description(f"Epoch {epoch + 1} - Accuracy: {epoch_accuracy:.4f} - Loss: {epoch_loss:.4f}")
            pbar.update(1)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = model.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    return model

# Testing function
def test(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels, keypoints in loader:
            outputs = model(images, keypoints).squeeze(1)
            loss = criterion(outputs, labels.float())

            total_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = correct / total
    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds))

# Train and test the model
model = train(model, train_loader, optimizer, criterion, n_epochs=10)
test(model, test_loader, criterion)
