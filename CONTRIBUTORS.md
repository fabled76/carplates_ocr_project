# Contributors

This file lists the contributors to the **Car Plates OCR Project**, along with their current tasks and responsibilities.

---

## 👤 Alex

**Tasks:**
- Dataset organisation and collection
- Organizing image data into appropriate directories (`train`, `val`, `test`)
- Ensuring dataset consistency and quality (file format, resolution, naming conventions)

**Next Steps:**
- Work on converting bounding box labels (if needed)
- Collaborate with Mariyam to labelling, splitting and validate the preprocessing pipeline on a small test set

---

## 👤 Mariyam

**Tasks:**
- Dataset labeling
- Image preprocessing (grayscale conversion, resizing, noise reduction)
- Implementation of data augmentation methods (rotation, scaling, contrast) for experiments
- Testing the effect of preprocessing on detection quality

**Next Steps:**
- Tune preprocessing parameters (e.g., thresholds for binarization or filters)
- Analyze how preprocessing affects each detection method (Canny, MSER, etc.)
- Document best practices and results for the pipeline

---

## Collaboration Notes

- Please commit to separate branches when working on major features.
- Use clear commit messages (e.g., `feat(preprocessing): added histogram equalization` or `fix X bubliki kroliki`)
- Submit pull requests for review before merging to `main`.

---
