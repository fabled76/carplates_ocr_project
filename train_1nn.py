import os
import cv2
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from evaluation import evaluate_model

# === 1. Zoning Feature Extractor ===
def extract_zoning_features(img, zones=(4, 4)):
    """
    Divide image into zones and compute density in each.
    Returns a 1D feature vector.
    """
    h, w = img.shape
    zh, zw = h // zones[0], w // zones[1]
    features = []

    for i in range(zones[0]):
        for j in range(zones[1]):
            zone = img[i * zh:(i + 1) * zh, j * zw:(j + 1) * zw]
            density = np.sum(zone == 255) / zone.size  # white pixel ratio
            features.append(density)

    return features

# === 2. Preprocessing Pipeline ===
def preprocess_image(img_path, size=(32, 32)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Binarize using Otsu
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize
    resized = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)

    return resized

# === 3. Load Dataset ===
def load_dataset(folder):
    X, y = [], []
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_path, file)
                processed = preprocess_image(img_path)
                if processed is not None:
                    features = extract_zoning_features(processed)
                    X.append(features)
                    y.append(class_name)
    return np.array(X), np.array(y)

# === 4. Load and Split ===
dataset_folder = 'data_chars'  # <- your root folder with subfolders per class
X, y = load_dataset(dataset_folder)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Train 1-NN Classifier ===
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# === 6. Predict & Evaluate ===
y_pred = knn.predict(X_test)

joblib.dump(knn, 'ocr_knn_model.pkl')

evaluate_model(y_test, y_pred)
