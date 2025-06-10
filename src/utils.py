import cv2
import os
import numpy as np

def preprocess_image(img_path, size=(32, 32), resize=True):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Binarize using Otsu
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    processed = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)

    return processed if resize else thresh

def load_dataset(folder, resize=True, size=(32, 32)):
    """
    Load images from subdirectories in the dataset folder.
    Each subdirectory name is used as a class label.
    """
    X, y, class_names = [], [], []
    for class_name in sorted(os.listdir(folder)):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        class_names.append(class_name)
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_path, file)
                processed = preprocess_image(img_path, size=size, resize=resize)
                if processed is not None:
                    features = extract_zoning_features(processed)  # Fixed-length features
                    if features is not None:
                        X.append(features)
                        y.append(class_name)
    return np.array(X), np.array(y), class_names


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
