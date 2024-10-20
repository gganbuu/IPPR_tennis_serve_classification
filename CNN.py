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
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

# Load datasets (assuming these functions are defined in Load_dataset.py)
from Load_dataset import load_datasets, load_test_datasets

# Define the base path for the dataset
base_path = 'datasets/serveDataset/'  # Set the correct path

# Load datasets
images, labels, keypoints = load_datasets(base_path)
test_images, test_labels, test_keypoints = load_test_datasets(base_path)

def rotate_image_and_keypoints(image, keypoints, angle):
    """
    Rotate both the image and the corresponding keypoints by the given angle.

    Args:
        image (Tensor): Image tensor to rotate.
        keypoints (Tensor): Tensor of shape (n, 2) containing (x, y) coordinates of keypoints.
        angle (float): Angle to rotate (in degrees).

    Returns:
        rotated_image (Tensor): The rotated image tensor.
        rotated_keypoints (Tensor): The transformed keypoints tensor.
    """
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
def prepare_dataset(images, labels, keypoints):
    image_tensors, keypoint_tensors = [], []
    label_list = []

    for img, kp, lab in zip(images, keypoints, labels):
        # Convert the original image and keypoints to tensors
        original_img_tensor = torch.tensor(img).permute(2, 0, 1)
        original_kp_tensor = torch.tensor(kp, dtype=torch.float32)

        # Append the original image and keypoints
        image_tensors.append(original_img_tensor)
        keypoint_tensors.append(original_kp_tensor)
        label_list.append(lab)  # Append the original label

        # If the label is 1, also apply transformations and store them
        if lab == 1:
            for x in range(3):
                transformed_img, transformed_kp = apply_transformations(
                    original_img_tensor.clone(), original_kp_tensor.clone()
                )
                # Append the transformed image and keypoints
                image_tensors.append(transformed_img)
                keypoint_tensors.append(transformed_kp)
                label_list.append(lab)  # Same label for the transformed version

    # Convert lists to stacked tensors
    images_tensor = torch.stack(image_tensors)
    keypoints_tensor = torch.stack(keypoint_tensors)
    labels_tensor = torch.tensor(label_list, dtype=torch.float32)

    # Print shapes for debugging
    print(f"Images shape: {images_tensor.shape}")
    print(f"Labels shape: {labels_tensor.shape}")
    print(f"Keypoints shape: {keypoints_tensor.shape}")

    return images_tensor, labels_tensor, keypoints_tensor

train_images_tensor, train_labels_tensor, train_keypoints_tensor = prepare_dataset(images, labels, keypoints)
test_images_tensor, test_labels_tensor, test_keypoints_tensor = prepare_dataset(test_images, test_labels, test_keypoints)

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
        self.fc2 = nn.Linear(64, 1)  # Output layer
        self.dropout = nn.Dropout(p=0.5)  # Reduced dropout rate

    def forward(self, x, keypoints):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))  # Convolution + Leaky ReLU + Pooling
        x = x.view(-1, 16 * 112 * 112)  # Flattening the tensor; this will be (N, 200704)

        # Flatten the keypoints and concatenate with the flattened features
        keypoints_flat = keypoints.view(keypoints.size(0), -1)  # Flatten keypoints to (N, 14)
        x = torch.cat((x, keypoints_flat), dim=1)  # Concatenate along the feature dimension
        
        x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.01))  # Fully connected layer with Leaky ReLU
        x = self.fc2(x)  # Output layer without activation (for BCEWithLogitsLoss)
        return x  # No sigmoid applied here
# Instantiate the model, loss function, and optimizer
model = ServeCNN()
class_weights = torch.tensor([1.0, (191 / 200)]).float()
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

# Training function
def train(model, loader, optimizer, criterion, n_epochs=1, patience=5):
    best_accuracy = 0.0
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None
    losses_bits = []  # Track losses

    model.train()
    with tqdm(total=n_epochs, unit="epoch") as pbar:
        for epoch in range(n_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in loader:  # Iterate over the loader
                images, labels, keypoints = batch  # Unpack all three elements

                # Forward pass
                outputs = model(images, keypoints).squeeze(1)  # Ensure correct shape for BCE loss
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

            # Update tqdm description
            pbar.set_description(f"Epoch {epoch + 1} - Accuracy: {epoch_accuracy:.4f} - Loss: {epoch_loss:.4f}")
            pbar.update(1)

            # Check for improvements
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
                best_model = model.state_dict().copy()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = model.state_dict().copy()

    print(f"Best training accuracy: {best_accuracy:.4f}")
    model.load_state_dict(best_model)  # Load the best model state
    return model, losses_bits

# Testing function
def test(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for images, labels, keypoints in loader:
            outputs = model(images, keypoints).squeeze(1)
            loss = criterion(outputs, labels.float())

            total_loss += loss.item()
            preds = (outputs >= 0.5).float()
            predictions.extend(preds.numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, predictions

# Train the model
model, losses_bits = train(model, train_loader, optimizer, criterion, n_epochs=2, patience=5)

# Test the model
test_loss, test_accuracy, test_predictions = test(model, test_loader, criterion)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Generate classification report
test_labels_np = test_labels_tensor.numpy()
print(classification_report(test_labels_np, test_predictions))


