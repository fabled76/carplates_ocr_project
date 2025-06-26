import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from src.evaluation import evaluate_model
from src.utils import extract_hog_features

# === Preprocessing Pipeline ===
def preprocess_image(img_path, size=(32, 32)):
    """Reads an image, converts to grayscale, and applies Otsu threshold."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)
    return resized   

# === Load Dataset (only "0" and "1") ===
def load_dataset(folder, allowed_classes=("0", "1")):
    """Loads only selected classes from the dataset directory."""
    X, y, class_names = [], [], []
    for class_name in sorted(os.listdir(folder)):
        if class_name not in allowed_classes:
            continue
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        class_names.append(class_name)
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_path, file)
                processed = preprocess_image(img_path)
                if processed is not None:
                    features = extract_hog_features(processed)
                    X.append(features)
                    y.append(class_name)
    return np.array(X), np.array(y), class_names

# === Main Execution ===
dataset_folder = 'data_chars'
X, y, class_names = load_dataset(dataset_folder)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA keeping 95% variance
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

# Split original PCA data for grid search
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Grid Search for best parameters on original PCA data
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto']
}
svc = SVC(probability=True)
grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Evaluate best model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# === Prepare 2D data for visualization ===
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.2, random_state=42)

# Parameter ranges for visualization
C_range = [0.1, 1, 10]
gamma_range = [0.001, 0.1, 1]

# Train classifiers for each param combo
classifiers = []
for C in C_range:
    for gamma in gamma_range:
        clf = SVC(C=C, kernel='rbf', gamma=gamma)
        clf.fit(X_train_2d, y_train_2d)
        classifiers.append((C, gamma, clf))

# Plot decision boundaries
plt.figure(figsize=(12, 9))

xx, yy = np.meshgrid(
    np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 200),
    np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 200)
)

for k, (C, gamma, clf) in enumerate(classifiers):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(len(C_range), len(gamma_range), k + 1)
    plt.title(f"gamma={gamma}, C={C}", fontsize=10)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)

    # Plot training samples with circles
    plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_2d.astype(int),
                cmap=plt.cm.RdBu, edgecolors="k", s=30, label='Train samples')

    # Plot test samples with stars
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d.astype(int),
                cmap=plt.cm.RdBu, edgecolors="k", s=100, marker='*', label='Test samples')

    plt.xticks([])
    plt.yticks([])

# Create legend handles for classes (colors)
class0_patch = mpatches.Patch(color='blue', label='Class 0')
class1_patch = mpatches.Patch(color='red', label='Class 1')

# Create legend handles for train/test markers (shapes)
train_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=6, label='Train samples')
test_marker = mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=12, label='Test samples')

plt.legend(handles=[class0_patch, class1_patch, train_marker, test_marker], loc='lower right', fontsize=9)

plt.suptitle("Decision Boundaries for Different C and Gamma (RBF)", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
