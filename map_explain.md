# Evaluation of mAP@0.5 and PR Curves for Tree Species Detection

This script calculates the **mean Average Precision at IoU threshold 0.5 (mAP@0.5)** and plots **precision-recall curves** for each tree species class using ground-truth and predicted bounding boxes. It is intended to evaluate the final detection performance of trained Faster R-CNN models on validation datasets.

---

## 🎯 Objective

To provide a fine-grained evaluation of object detection performance across individual classes using:
- Per-class **Precision-Recall (PR) curves**
- **Average Precision (AP)** for each class
- **mAP@0.5** as an overall metric

---

## 🔍 Methodology

1. **Load Ground Truth and Predictions**
   - Ground-truth boxes are read from `val_labels.csv`
   - Predicted boxes and scores are read from `merged_predictions.csv` (output of the final model)

2. **Per-Class Evaluation**
   - For each class label:
     - Ground-truth boxes are grouped by image
     - Predictions are sorted by confidence scores (descending)
     - Detections are matched to ground-truth boxes using IoU ≥ 0.5

3. **Precision-Recall Computation**
   - True positives and false positives are accumulated
   - Precision and recall arrays are calculated
   - AP for each class is computed using 11-point interpolation

4. **Final Result**
   - mAP@0.5 is computed as the mean of APs across all classes
   - Precision-Recall curves are visualized for each class

---

## 📁 Inputs

- `val_labels.csv`: Validation set with ground-truth bounding boxes  
- `merged_predictions.csv`: Model predictions with columns:
