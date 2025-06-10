import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from src.vis_utils import (
    compute_average_image_per_class,
    compute_variance_image_per_class,
    plot_image_count_per_class,
    plot_resolution_aspect_ratio,
    plot_image_overview
)

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
        if f.lower().endswith('.jpg')
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

# creating DataFrame
df = pd.DataFrame(data)

# creating plots
compute_average_image_per_class(dataset_dir, output_dir)
compute_variance_image_per_class(dataset_dir, output_dir)
plot_image_count_per_class(image_counts, output_dir)
plot_resolution_aspect_ratio(dataset_dir, output_dir)
plot_image_overview(dataset_dir, output_dir)
