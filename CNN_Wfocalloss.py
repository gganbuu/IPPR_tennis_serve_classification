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

from Load_dataset import load_datasets, load_test_datasets  # Load datasets from custom module

# Define dataset paths
base_path = 'datasets/serveDataset/'  # Adjust as needed

# Load datasets
images, labels, keypoints = load_datasets(base_path)
test_images, test_labels, test_keypoints = load_test_datasets(base_path)

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of correct classification
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def rotate_image_and_keypoints(image, keypoints, angle):
    rotated_image = TF.rotate(image, angle)
    theta = math.radians(angle)

    Cx, Cy = image.shape[2] / 2, image.shape[1] / 2
    rotation_matrix = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])

    shifted_keypoints = keypoints - torch.tensor([Cx, Cy])
    rotated_keypoints = (rotation_matrix @ shifted_keypoints.T).T + torch.tensor([Cx, Cy])

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
        

    return torch.stack(image_tensors), torch.tensor(label_list, dtype=torch.float32), torch.stack(keypoint_tensors)

# Prepare datasets and DataLoaders
train_images_tensor, train_labels_tensor, train_keypoints_tensor = prepare_dataset(images, labels, keypoints)
test_images_tensor, test_labels_tensor, test_keypoints_tensor = prepare_dataset(test_images, test_labels, test_keypoints)


train_dataset = TensorDataset(train_images_tensor, train_labels_tensor, train_keypoints_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(test_images_tensor, test_labels_tensor, test_keypoints_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class ServeCNN(nn.Module):
    def __init__(self):
        super(ServeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 112 * 112 + 14, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, keypoints):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = x.view(-1, 16 * 112 * 112)

        keypoints_flat = keypoints.view(keypoints.size(0), -1)
        x = torch.cat((x, keypoints_flat), dim=1)

        x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.01))
        return torch.sigmoid(self.fc2(x))

model = ServeCNN()
criterion = FocalLoss(alpha=0.25, gamma=2.0)  # Use Focal Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
# Initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(initialize_weights)



# Modified training loop with scheduler
def train(model, loader, optimizer, criterion, scheduler, n_epochs=10, patience=3):
    best_model, best_loss, best_accuracy = None, float('inf'), 0.0
    epochs_without_improvement = 0

    model.train()
    with tqdm(total=n_epochs, unit="epoch") as pbar:
        for epoch in range(n_epochs):
            total_loss, correct, total = 0.0, 0, 0

            for images, labels, keypoints in loader:
                outputs = model(images, keypoints).squeeze(1)
                loss = criterion(outputs, labels)

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

            scheduler.step()  # Update learning rate

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


def test(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels, keypoints in loader:
            outputs = model(images, keypoints).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {correct / total:.4f}")
    print(classification_report(all_labels, all_preds))
# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model = train(model, train_loader, optimizer, criterion, n_epochs=1,scheduler=scheduler)
test(model, test_loader, criterion)
