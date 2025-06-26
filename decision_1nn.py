import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def extract_zone_feature(img, zone_index, zones=(4, 4)):
    """Extract normalized density of a specific zone."""
    h, w = img.shape
    zh, zw = h // zones[0], w // zones[1]
    i, j = zone_index
    zone = img[i * zh:(i + 1) * zh, j * zw:(j + 1) * zw]
    return np.sum(zone == 255) / zone.size

def load_data(folder, classes=("F", "E", "H")):
    X, y = [], []
    for class_name in classes:
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir): continue
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img = cv2.imread(os.path.join(class_dir, file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                img = cv2.resize(thresh, (32, 32))
                
                # Features
                f_bottom_right = extract_zone_feature(img, zone_index=(3, 1))
                f_bottommost = extract_zone_feature(img, zone_index=(3, 2))
                
                X.append([f_bottom_right, f_bottommost])
                y.append(class_name)

    return np.array(X), np.array(y)

dataset_folder = 'data_chars'  # Adjust path
X, y = load_data(dataset_folder)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Pick random test sample
test_index = random.randrange(len(X_test))
test_sample = X_test[test_index].reshape(1, -1)
test_label = y_test[test_index]
pred_label = knn.predict(test_sample)[0]

distances, indices = knn.kneighbors(test_sample)

mask_F = (y_train == 'F')
mask_E = (y_train == 'E')
mask_H = (y_train == 'H')
plt.figure(figsize=(10, 6))

# Training data
plt.scatter(X_train[mask_F][:, 0], X_train[mask_F][:, 1], c='blue', label='Train F', alpha=0.5)
plt.scatter(X_train[mask_E][:, 0], X_train[mask_E][:, 1], c='green', label='Train E', alpha=0.5)
plt.scatter(X_train[mask_H][:, 0], X_train[mask_H][:, 1], c='purple', label='Train H', alpha=0.5)

# Test sample
plt.scatter(test_sample[0, 0], test_sample[0, 1],
            c='red', marker='x', s=150, label=f'Test ({test_label}) -> Pred: {pred_label}')

# Nearest neighbor
neighbor_point = X_train[indices.flatten()]
plt.scatter(neighbor_point[:, 0], neighbor_point[:, 1],
            c='black', s=100, edgecolors='yellow',
            label='Nearest Neighbor')

# Labels
plt.xlabel("Feature 1: Bottom-Right1 Zone Density")
plt.ylabel("Feature 2: Bottom-Right2 Zone Density")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.title(f"2D Feature Space for 'F', 'E', 'H' | Test = {test_label}, Pred = {pred_label}")

plt.show()
