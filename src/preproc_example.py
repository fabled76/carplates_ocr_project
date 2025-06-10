import cv2
import matplotlib.pyplot as plt


def preprocess_image(img_path, size=(32, 32)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Binarize using Otsu
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize
    processed = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)

    return img, processed

def visualize_preprocessing(img_path, size=(32, 32)):
    original, processed = preprocess_image(img_path, size=size)

    plt.figure(figsize=(8, 4))

    # original
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # processed
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(f'Preprocessed (Resized to {size})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

img_path = '../data_chars/7/1xemay362_7.jpg'
visualize_preprocessing(img_path, size=(32, 32))
