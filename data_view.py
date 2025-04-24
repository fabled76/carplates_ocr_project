import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy

# Dataset path
dataset_dir = 'data_chars'

data = []
image_counts = {}

# === Extract Data ===
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

# === 1. Bar Chart: Image Count per Class ===
plt.figure(figsize=(10, 5))
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
plt.show()

# === 2. Correlogram: Correlation of Class-wise Features ===
grouped = df.groupby('class').mean(numeric_only=True)
corr = grouped.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlogram of Class-wise Grayscale Feature Averages')
plt.tight_layout()
plt.show()
