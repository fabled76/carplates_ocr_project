import cv2
import numpy as np
import joblib
from skimage.feature import hog
import matplotlib.pyplot as plt

# Load the trained SVM model and label encoder
model_path = "svm_ocr_model.pkl"
label_encoder_path = "label_encoder.pkl"
svm = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

def preprocess_image(image):
    return image

def extract_hog_features(image):
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    hog_features = hog(resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]

    iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-5)
    return iou

def filter_boxes_by_iou(boxes, iou_threshold=0.1):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)  # larger first
    filtered = []

    for box in boxes:
        keep = True
        for kept in filtered:
            if compute_iou(box, kept) > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(box)

    return filtered

def detect_regions_canny(image):
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 10 < w < 100 and 10 < h < 100:
            boxes.append((x, y, w, h))

    print(f"[Canny] Detected {len(boxes)} components before IoU filtering")

    boxes = filter_boxes_by_iou(boxes)
    print(f"[Canny] Remaining {len(boxes)} components after IoU filtering")

    return sorted(boxes, key=lambda b: b[0])

def detect_regions_selective_search(image, show_debug=True):
    print("[Step 1] Initializing Selective Search...")

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    base_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ss.setBaseImage(base_image)

    print("[Step 2] Switching to Fast Mode...")
    ss.switchToSelectiveSearchQuality()

    print("[Step 3] Running Selective Search...")
    rects = ss.process()
    print(f"[Step 4] Total Regions Proposed: {len(rects)}")

    boxes = []
    for (x, y, w, h) in rects:
        if 10 < w < 100 and 10 < h < 100:
            boxes.append((x, y, w, h))

    print(f"[Step 5] Regions after size filter: {len(boxes)}")

    if show_debug:
        debug_image = base_image.copy()
        for (x, y, w, h) in boxes[:50]:
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB))
        plt.title("Top 50 Selective Search Proposals (after size filter)")
        plt.axis('off')
        plt.show()

    boxes = filter_boxes_by_iou(boxes)

    print(f"[Step 6] Remaining Boxes after IoU filter: {len(boxes)}")

    return sorted(boxes, key=lambda b: b[0])

def detect_regions_mser(image):
    print("[MSER] Initializing MSER detector...")
    mser = cv2.MSER_create()
    mser.setMinArea(100)
    mser.setMaxArea(10000)

    print("[MSER] Detecting regions...")
    regions, _ = mser.detectRegions(image)

    boxes = []
    for p in regions:
        x, y, w, h = cv2.boundingRect(p.reshape(-1, 1, 2))
        if 10 < w < 100 and 10 < h < 100:
            boxes.append((x, y, w, h))

    print(f"[MSER] Detected {len(boxes)} components before IoU filtering")
    boxes = filter_boxes_by_iou(boxes)
    print(f"[MSER] Remaining {len(boxes)} components after IoU filtering")

    return sorted(boxes, key=lambda b: b[0])

def detect_regions_cca(image):
    print("[CCA] Performing Connected Component Analysis...")
    
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for i in range(1, num_labels):  # skip the background
        x, y, w, h, area = stats[i]
        if 10 < w < 100 and 10 < h < 100:
            boxes.append((x, y, w, h))

    print(f"[CCA] Detected {len(boxes)} components before IoU filtering")
    boxes = filter_boxes_by_iou(boxes)
    print(f"[CCA] Remaining {len(boxes)} components after IoU filtering")

    return sorted(boxes, key=lambda b: b[0])


def predict(image_path, method="canny"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.imread(image_path)
    image = preprocess_image(image)

    if method == "canny":
        components = detect_regions_canny(image)
    elif method == "selective_search":
        components = detect_regions_selective_search(image)
    elif method == "mser":
        components = detect_regions_mser(image)
    elif method == "cca":
        components = detect_regions_cca(image)
    else:
        raise ValueError("Invalid method. Use 'canny', 'selective_search', or 'mser'.")

    print(f"Detected Components: {len(components)}")

    if len(components) == 0:
        print("No character regions detected.")
        return "", original_image, []

    predicted_labels = []
    for (x, y, w, h) in components:
        char_region = image[y:y+h, x:x+w]
        features = extract_hog_features(char_region)
        features = features.reshape(1, -1)
        predicted_char = label_encoder.inverse_transform(svm.predict(features))[0]
        predicted_labels.append(predicted_char)

    return ''.join(predicted_labels), original_image, components

if __name__ == "__main__":
    test_image_path = "image.png"

    detection_method = "cca" # Change to "canny", "selective_search", "mser" or "cca" as needed

    predicted_text, original_image_with_boxes, components = predict(test_image_path, method=detection_method)

    print(f"Predicted Text: {predicted_text}")

    for (x, y, w, h) in components:
        cv2.rectangle(original_image_with_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(original_image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title(f'Bounding Boxes Over Characters ({detection_method.capitalize()} + IoU)')
    plt.axis('off')
    plt.show()
