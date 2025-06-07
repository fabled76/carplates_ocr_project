import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy

# Dataset path and output directory for plots
dataset_dir = 'data_chars'
output_dir = 'plots'

data = []
image_counts = {}

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]
    image_counts[class_name] = len(image_files)

    for filename in image_files:
        file_path = os.path.join(class_path, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        flat = img.flatten()

        feature_vector = {
            'mean': np.mean(flat),
            'std': np.std(flat),
            'min': np.min(flat),
            'max': np.max(flat),
            'median': np.median(flat),
            'skewness': skew(flat),
            'kurtosis': kurtosis(flat),
            'entropy': shannon_entropy(img),
            'class': class_name
        }

        data.append(feature_vector)

# Create DataFrame
df = pd.DataFrame(data)

# === Image Count per Class ===
plt.figure(figsize=(13, 9))
bars = plt.bar(image_counts.keys(), image_counts.values(), color='lightblue')

total_images = sum(image_counts.values())
for bar in bars:
    height = bar.get_height()
    percent = (height / total_images) * 100
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
             f'{percent:.1f}%', ha='center', va='bottom')

plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Image Count per Class with Percentages')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'image_count_per_class.png'), dpi=300, bbox_inches='tight')
plt.show()

# === Image resolution distribution and aspect ratio distribution ===
resolutions = []
aspect_ratios = []

for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            file_path = os.path.join(class_path, filename)
            img = cv2.imread(file_path)
            if img is not None:
                h, w = img.shape[:2]
                resolutions.append((w, h))
                aspect_ratios.append(w / h)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(*zip(*resolutions), alpha=0.5, color='teal', s=50, edgecolor='black')
plt.xlabel('Width (pixels)', fontweight='bold')
plt.ylabel('Height (pixels)', fontweight='bold')
plt.title('Image Resolution Distribution', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
sns.histplot(aspect_ratios, bins=30, kde=True, color='coral', edgecolor='black')
plt.xlabel('Aspect Ratio (Width/Height)', fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.title('Aspect Ratio Distribution', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'resolution_aspect_ratio.png'), dpi=300, bbox_inches='tight')
plt.show()
