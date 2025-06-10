import os
import cv2

# Paths
images_dir = '../data/images/train'
labels_dir = '../data/labels/train'
output_dir = '../data_chars/'

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over image files
for image_filename in os.listdir(images_dir):
    if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(images_dir, image_filename)
    label_path = os.path.join(labels_dir, os.path.splitext(image_filename)[0] + '.txt')

    # Skip if label file doesn't exist
    if not os.path.exists(label_path):
        continue

    # Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip malformed lines

            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)

            # Convert YOLO format to pixel coordinates
            x_center *= w
            y_center *= h
            width *= w
            height *= h

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Crop and save
            cropped = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            class_folder = os.path.join(output_dir, str(class_id))
            os.makedirs(class_folder, exist_ok=True)

            crop_filename = f"{os.path.splitext(image_filename)[0]}_{i}.jpg"
            cv2.imwrite(os.path.join(class_folder, crop_filename), cropped)

print("Cropping complete.")
