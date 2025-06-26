import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import extract_hog_features

def preprocess_image(img_path, size=(32, 32)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)
    return resized

def load_dataset(folder, selected_classes=("0", "1")):
    X, y = [], []
    for class_name in sorted(os.listdir(folder)):
        if class_name not in selected_classes:
            continue
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_path, file)
                processed = preprocess_image(img_path)
                if processed is not None:
                    features = extract_hog_features(processed)
                    X.append(features)
                    y.append(int(class_name))  # convert labels to int for easy coloring
    return np.array(X), np.array(y)

dataset_folder = 'data_chars'
X, y = load_dataset(dataset_folder, selected_classes=("0", "1"))

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42, stratify=y)

# Train Linear SVM
svm_model = SVC(C=1, kernel='linear')
svm_model.fit(X_train, y_train)

# Prepare meshgrid for decision boundary
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))

# Plot decision boundary with light fill
plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)

# Plot training points
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', edgecolors='k', label='Train Class 0', alpha=0.7, cmap=plt.cm.RdBu)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', edgecolors='k', label='Train Class 1', alpha=0.7, cmap=plt.cm.RdBu)

# Plot test points with star markers
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='blue', edgecolors='k', marker='*', s=150, label='Test Class 0', cmap=plt.cm.RdBu)
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='red', edgecolors='k', marker='*', s=150, label='Test Class 1', cmap=plt.cm.RdBu)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Linear SVM Decision Boundary on 2D PCA of HOG Features")

# Custom legend for classes and train/test
class0_patch = mpatches.Patch(color='blue', label='Class 0')
class1_patch = mpatches.Patch(color='red', label='Class 1')
train_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Train samples')
test_marker = mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=12, label='Test samples')

plt.legend(handles=[class0_patch, class1_patch, train_marker, test_marker], loc='lower right')

plt.tight_layout()
plt.show()
