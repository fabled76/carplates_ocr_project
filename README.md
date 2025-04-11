# Car Plates OCR Project

This project implements an Optical Character Recognition (OCR) pipeline for detecting and recognizing characters on car license plates. The goal is to segment and classify each symbol from plate images using classical computer vision and machine learning techniques.

## Features

- **Character Region Detection**:
  - **Canny Edge Detection**
  - **MSER (Maximally Stable Extremal Regions)**
  - **Selective Search**
  - **Connected Component Analysis (CCA)**
  
- **Feature Extraction**:
  - **Histogram of Oriented Gradients (HOG)** for extracting features from segmented regions.
  
- **Character Classification**:
  - **Support Vector Machine (SVM)** for classifying characters based on HOG features.
  
- **Bounding Box Filtering**:
  - Non-maximum suppression based on Intersection over Union (IoU) to remove overlapping boxes and keep the most relevant detections.

## Tech Stack

- **Python**
- **OpenCV** for image processing
- **scikit-learn** for machine learning and SVM classification
- **scikit-image** for feature extraction and region detection
- **matplotlib** for visualization
- **joblib** for model serialization

## Project Structure

├── svm_pipeline.py # Main pipeline for OCR ├── svm_ocr_model.pkl # Trained SVM model ├── label_encoder.pkl # Label encoder for characters ├── datasets/ │ └── test/ │ └── images/ # Test images of car plates


### `svm_pipeline.py`
The core pipeline for OCR processing. It detects character regions using different detection methods, extracts HOG features, and then classifies the regions using a pre-trained SVM model.

### `svm_ocr_model.pkl`
This is the trained SVM model that is used for character classification.

### `label_encoder.pkl`
The label encoder used to map numeric predictions back to character labels.

### `datasets/test/images/`
A directory where test images of car license plates are stored for testing the pipeline.

## Getting Started

To get started with the project, follow these steps:

### Prerequisites

1. Install the required dependencies:

   ```bash
   pip install opencv-python-headless scikit-learn scikit-image matplotlib joblib
   ```

2. Download or clone this repository to your local machine.

  ```bash
  git clone https://github.com/yourusername/carplates_ocr_project.git
  cd carplates_ocr_project
  ```
3. Running the OCR Pipeline
Modify the detection method in the script svm_pipeline.py (select one of "canny", "selective_search", "mser", or "cca").

Run the OCR pipeline on a test image:

  ```bash
  python svm_pipeline.py
  View the results and the predicted text on the license plate.
  ```

## Example Output
After running the script, the predicted text for the license plate is printed in the terminal, and the processed image with bounding boxes around detected characters is displayed.

## Methods
 - Canny Edge Detection
Uses Canny to detect edges in the image and find contours to identify potential character regions.

 - MSER (Maximally Stable Extremal Regions)
MSER is a region-based segmentation method, ideal for detecting regions that represent characters in license plates.

Selective Search
Selective Search proposes possible bounding boxes for regions of interest and refines them through quality scoring.

Connected Component Analysis (CCA)
Identifies connected components in binary images and filters out irrelevant regions based on size and shape.

Notes
Image Quality: The best results are achieved when the license plates are clearly visible with high contrast and good alignment.

Performance: The performance can vary depending on the detection method selected. You may need to experiment with different methods to get the best results for your specific dataset.

Contributing
