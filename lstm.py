import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

# Convert keypoints to LSTM-compatible format
# Assuming keypoints shape is (num_samples, num_timesteps, num_features)
def reshape_keypoints_for_lstm(keypoints):
    return keypoints.reshape(keypoints.shape[0], keypoints.shape[1], -1)  # Reshape to (N, T, F)

train_keypoints_lstm = reshape_keypoints_for_lstm(keypoints)
test_keypoints_lstm = reshape_keypoints_for_lstm(test_keypoints)

# Create DataLoaders
train_dataset = TensorDataset(torch.tensor(train_keypoints_lstm, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(torch.tensor(test_keypoints_lstm, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the LSTM model

class KeypointLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(KeypointLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Assuming binary classification

    def forward(self, x):
        out, _ = self.lstm(x)  # Get LSTM outputs
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)

        print(f"LSTM Output shape: {out.shape}")  # Debugging: Check output shape
        return out  # Remove squeeze() for debugging


# Instantiate the model, loss function, and optimizer
input_size = train_keypoints_lstm.shape[2]  # Number of features (e.g., 14 keypoints)
hidden_size = 64
num_layers = 2

model = KeypointLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Generate classification report and confusion matrix, then save them to a JSON file
def save_results_to_json(test_labels, test_predictions, train_accuracy, train_loss, test_accuracy, test_loss, output_path="classification_results.json"):
    # No need to convert test_labels to NumPy as it is already an array
    test_labels_np = test_labels  # Keep as is
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
        "test_loss": test_loss
    }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

# Train and test functions
def train(model, loader, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for keypoints, labels in tqdm(loader):
            optimizer.zero_grad()
            outputs = model(keypoints)  # Outputs shape should be [N, 1]
            loss = criterion(outputs, labels.view(-1, 1))  # Ensure labels are shaped [N, 1]
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5  # Convert to binary predictions
            correct += (predictions == labels.view(-1, 1)).sum().item()  # Ensure labels are shaped [N, 1]
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        avg_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return avg_accuracy, avg_loss

def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    predictions_list = []

    with torch.no_grad():
        for keypoints, labels in loader:
            outputs = model(keypoints)  # Outputs shape should be [N, 1]
            loss = criterion(outputs, labels.view(-1, 1))  # Ensure labels are shaped [N, 1]
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5  # Convert to binary predictions
            predictions_list.extend(predictions.cpu().numpy())
            correct += (predictions == labels.view(-1, 1)).sum().item()  # Ensure labels are shaped [N, 1]
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    avg_accuracy = correct / total
    return avg_accuracy, avg_loss, predictions_list

# Train the model
train_accuracy, train_loss = train(model, train_loader, optimizer, criterion, epochs=5)

# Test the model
test_accuracy, test_loss, predictions = test(model, test_loader, criterion)

# Save the results
save_results_to_json(test_labels, predictions, train_accuracy, train_loss, test_accuracy, test_loss)

