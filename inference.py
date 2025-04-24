import joblib
import cv2
import numpy as np

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

def preprocess_image(img_path, size=(32, 32)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Binarize using Otsu
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize
    resized = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)

    return resized

def predict_character(img_path, model_path='ocr_knn_model.pkl'):
    # Load model
    knn = joblib.load(model_path)

    # Preprocess image
    processed = preprocess_image(img_path)
    if processed is None:
        raise ValueError("Invalid image")

    # Extract features
    features = extract_zoning_features(processed)

    # Predict
    predicted_class = knn.predict([features])[0]
    return predicted_class

img_path = '.jpg'  # Example image path

print(predict_character(img_path))
