import os
import joblib
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from src.evaluation import evaluate_model
from src.utils import extract_hog_features

# === Preprocessing Pipeline ===
def preprocess_image(img_path, size=(32, 32)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)
    return resized

# === Load Dataset ===
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

# === Load and Preprocess Dataset ===
dataset_folder = 'data_chars'
X, y, class_names = load_dataset(dataset_folder)

# === Standardize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Define Parameter Grid ===
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 30],
}

# === Random Search ===
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
random_search.fit(X_train, y_train)

# === Evaluation ===
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best Parameters: {random_search.best_params_}")
evaluate_model(y_test, y_pred, class_names)

# === Save Models ===
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/ocr_rf_model.pkl')
joblib.dump(scaler, 'models/ocr_scaler.pkl')
