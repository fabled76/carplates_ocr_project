import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def compute_average_image_per_class(dataset_dir, output_dir, target_size=(128, 128)):
    """
    Compute pixel-wise average image for each class, resizing images to target_size.
    Images are saved in output_dir/mean_images in grayscale.

    Args:
        dataset_dir (str): Path to the dataset directory.
        output_dir (str): Path to save average images.
        target_size (tuple): (width, height) to resize images to (default: 128x128).
    """
    # Create subfolder for mean images
    mean_dir = os.path.join(output_dir, 'mean_images')
    if not os.path.exists(mean_dir):
        os.makedirs(mean_dir)

    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        image_arrays = []
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(class_path, filename)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize image to target size
                    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    image_arrays.append(img_resized.astype(np.float32))

        if not image_arrays:
            print(f"No valid images found for class {class_name}.")
            continue

        # Compute pixel-wise average
        sum_image = np.sum(image_arrays, axis=0)
        avg_image = sum_image / len(image_arrays)
        avg_image = avg_image.astype(np.uint8)  # Convert to uint8 for saving

        # Save the average image
        output_path = os.path.join(mean_dir, f'average_image_{class_name}.png')
        cv2.imwrite(output_path, avg_image)
        print(f"Average image for class {class_name} saved to {output_path}")


def compute_variance_image_per_class(dataset_dir, output_dir, target_size=(128, 128)):
    """
    Compute pixel-wise variance image for each class, resizing images to target_size.
    Images are saved in output_dir/variance_images in grayscale, normalized to [0, 255].

    Args:
        dataset_dir (str): Path to the dataset directory.
        output_dir (str): Path to save variance images.
        target_size (tuple): (width, height) to resize images to (default: 128x128).
    """
    # Create subfolder for variance images
    variance_dir = os.path.join(output_dir, 'variance_images')
    if not os.path.exists(variance_dir):
        os.makedirs(variance_dir)

    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        image_arrays = []
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(class_path, filename)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize image to target size
                    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    image_arrays.append(img_resized.astype(np.float32))

        if not image_arrays:
            print(f"No valid images found for class {class_name}.")
            continue

        # Stack images into a 3D array for variance computation
        image_stack = np.stack(image_arrays, axis=0)

        # Compute pixel-wise variance
        var_image = np.var(image_stack, axis=0)

        # Normalize variance to [0, 255] for visualization
        var_min = np.min(var_image)
        var_max = np.max(var_image)
        if var_max > var_min:  # Avoid division by zero
            var_image = (var_image - var_min) / (var_max - var_min) * 255
        else:
            var_image = np.zeros_like(var_image)  # If variance is constant, set to zero
        var_image = var_image.astype(np.uint8)

        # Save the variance image
        output_path = os.path.join(variance_dir, f'variance_image_{class_name}.png')
        cv2.imwrite(output_path, var_image)
        print(f"Variance image for class {class_name} saved to {output_path}")


def plot_image_count_per_class(image_counts, output_dir):
    """
    Generate a bar plot of image counts per class with percentage labels.

    Args:
        image_counts (dict): Dictionary of class names and their image counts.
        output_dir (str): Path to save the plot.
    """
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


def plot_resolution_aspect_ratio(dataset_dir, output_dir):
    """
    Generate scatter plot for image resolution and histogram for aspect ratio.

    Args:
        dataset_dir (str): Path to the dataset directory.
        output_dir (str): Path to save the plot.
    """
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


def plot_image_overview(dataset_dir, output_dir):
    """
    Generate an overview plot showing mean and variance images for each class.

    Args:
        dataset_dir (str): Path to the dataset directory.
        output_dir (str): Path to save the overview plot.
    """
    mean_dir = os.path.join(output_dir, 'mean_images')
    variance_dir = os.path.join(output_dir, 'variance_images')

    # Get list of classes
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not classes:
        print("No classes found in dataset directory.")
        return

    # Set up figure: 2 columns (mean, variance), one row per class
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, 2, figsize=(8, 4 * n_classes))

    # Handle single class case (axes is not a 2D array)
    if n_classes == 1:
        axes = np.array([axes]).reshape(1, -1)

    for i, class_name in enumerate(sorted(classes)):
        # Load mean image
        mean_path = os.path.join(mean_dir, f'average_image_{class_name}.png')
        mean_img = cv2.imread(mean_path, cv2.IMREAD_GRAYSCALE)

        # Load variance image
        variance_path = os.path.join(variance_dir, f'variance_image_{class_name}.png')
        variance_img = cv2.imread(variance_path, cv2.IMREAD_GRAYSCALE)

        # Plot mean image
        if mean_img is not None:
            axes[i, 0].imshow(mean_img, cmap='gray')
            axes[i, 0].set_title(f'{class_name} Mean')
        else:
            axes[i, 0].text(0.5, 0.5, 'No Mean Image', ha='center', va='center')
        axes[i, 0].axis('off')

        # Plot variance image
        if variance_img is not None:
            axes[i, 1].imshow(variance_img, cmap='gray')
            axes[i, 1].set_title(f'{class_name} Variance')
        else:
            axes[i, 1].text(0.5, 0.5, 'No Variance Image', ha='center', va='center')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()
