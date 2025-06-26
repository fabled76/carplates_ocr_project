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
from sklearn.metrics import accuracy_score
from src.utils import extract_hog_features

def preprocess_image(img_path, size=(32, 32)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)
    return resized

# === Load Dataset (only classes "0" and "1") ===
def load_dataset(folder, allowed_classes=("0", "1")):
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
                    y.append(int(class_name))
    return np.array(X), np.array(y), class_names

# === Main ===
dataset_folder = 'data_chars'
X, y, class_names = load_dataset(dataset_folder)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction to 2D for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Split for train/test
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.2, random_state=42)

# Define SVM kernels to compare
kernels = [
    ('Linear', 'linear'),
    ('Polynomial (degree 3)', 'poly'),
    ('RBF', 'rbf'),
    ('Sigmoid', 'sigmoid')
]

# Train classifiers
classifiers = []
for name, kernel in kernels:
    if kernel == 'poly':
        clf = SVC(kernel=kernel, degree=3, C=10)
    else:
        clf = SVC(kernel=kernel, C=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    classifiers.append((name, clf, acc))

# Create meshgrid for plotting decision boundaries
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

# Plot results
plt.figure(figsize=(12, 10))

for i, (name, clf, acc) in enumerate(classifiers, 1):
    plt.subplot(2, 2, i)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    scatter_train = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=30, edgecolors='k', label='Train')
    scatter_test = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=60, marker='*', edgecolors='k', label='Test')

    plt.title(f"{name} Kernel\nTest Accuracy: {acc:.2f}")
    plt.xticks([])
    plt.yticks([])

# Create legend for classes
class_0_patch = mpatches.Patch(color='blue', label='Class 0')
class_1_patch = mpatches.Patch(color='red', label='Class 1')

# Create legend for train/test markers
train_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=8, label='Train samples')
test_marker = mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=12, label='Test samples')

plt.legend(handles=[class_0_patch, class_1_patch, train_marker, test_marker], loc='lower right')

plt.suptitle("SVM Kernel Comparison on 2D PCA of HOG Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
