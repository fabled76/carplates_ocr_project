import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from evaluation import evaluate_model

# === 1. HOG Feature Extractor ===
def extract_hog_features(img, size=(32, 32)):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        _winSize=size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    return hog.compute(img).flatten()

# === 2. Preprocessing Pipeline ===
def preprocess_image(img_path, size=(32, 32)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# === 3. Load Dataset ===
def load_dataset(folder):
    X, y, class_names = [], [], []
    for class_name in sorted(os.listdir(folder)):
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

# === 4. Load and Preprocess Dataset ===
dataset_folder = 'data_chars'
X, y, class_names = load_dataset(dataset_folder)

# === 5. Standardize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 6. Apply PCA ===
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

# === 7. Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# === 8. Define Parameter Grid ===
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']  # relevant only for 'rbf'
}

# === 9. Grid Search ===
svc = SVC(probability=True)
grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# === 10. Evaluation ===
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best Parameters: {grid_search.best_params_}")
evaluate_model(y_test, y_pred, class_names)

# === 11. Save Models ===
joblib.dump(best_model, 'ocr_svm_model.pkl')
joblib.dump(scaler, 'ocr_scaler.pkl')
joblib.dump(pca, 'ocr_pca.pkl')
