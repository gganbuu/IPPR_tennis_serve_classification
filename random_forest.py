import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Set the directory - Adjust accordingly
train_good_dir = '/Users/jiheenoh/Desktop/Tennis_serve/Train_good'
train_bad_dir = '/Users/jiheenoh/Desktop/Tennis_serve/Train_bad'

# Define augmentation
def load_images_from_folder(folder, label, augment=False):
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')
    file_count = 0
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
                    labels.append(label)
                    file_count += 1

                    if augment:
                        augmented_images = augment_image(img)
                        images.extend(augmented_images)
                        labels.extend([label] * len(augmented_images))
    print(f"Loaded {file_count} images from {folder}", flush=True)
    return np.array(images), np.array(labels)

# Image augementation
def augment_image(img):
    rows, cols, _ = img.shape
    augmented_images = []
    # Add rotation
    for angle in [10, -10, 20, -20]:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rotated = cv2.warpAffine(img, M, (cols, rows))
        augmented_images.append(img_rotated)
    # Adjust bright
    img_brighter = cv2.convertScaleAbs(img, alpha=1.3, beta=40)
    img_darker = cv2.convertScaleAbs(img, alpha=0.7, beta=-40)
    # Add noise
    noise = np.random.randint(0, 50, img.shape, dtype='uint8')
    img_noisy = cv2.add(img, noise)
    augmented_images.extend([img_brighter, img_darker, img_noisy])
    return augmented_images

# Load images from each folder
good_images, good_labels = load_images_from_folder(train_good_dir, 'good')
bad_images, bad_labels = load_images_from_folder(train_bad_dir, 'bad', augment=True)

# Set the num of 'bad' images evenly with 'good' images (Random sampling)
if len(bad_images) > len(good_images):
    indices = np.random.choice(len(bad_images), size=len(good_images), replace=False)
    bad_images = bad_images[indices]
    bad_labels = bad_labels[indices]

# Check data
print(f"Number of good images: {len(good_images)}", flush=True)
print(f"Number of bad images (balanced): {len(bad_images)}", flush=True)

# Merge data
X = np.concatenate([good_images, bad_images])
y = np.concatenate([good_labels, bad_labels])
print(f"Total number of images: {len(X)}", flush=True)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,               # Decrease to reduce overfitting
    min_samples_split=20,      # Increase to reduce overfitting
    min_samples_leaf=10,       # Increase to reduce overfitting
    class_weight='balanced',
    random_state=42
)

# Flattened the data
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)
print("Training model...", flush=True)
model.fit(X_train_flat, y_train)
print("Model training complete", flush=True)

# Cross validation
cv_scores = cross_val_score(model, X_train_flat, y_train, cv=5)
plt.figure(figsize=(10, 5))
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-')
plt.title("Cross-Validation Accuracy Scores")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0) 
plt.show()
print("Cross-Validation Accuracy Scores:", cv_scores, flush=True)
print("Mean CV Accuracy:", np.mean(cv_scores), flush=True)

y_pred = model.predict(X_test_flat)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
