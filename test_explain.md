# Evaluation Pipeline for Tree Species Detection using Faster R-CNN

This repository contains an evaluation pipeline for analyzing the performance of self-trained Faster R-CNN models in the context of tree species detection. The models have been trained using pseudo-labeling at a fixed confidence threshold (e.g., 0.95), and this script assesses their accuracy on a common validation set.

---

## 🎯 Purpose

To systematically evaluate model predictions across multiple training runs and extract key performance metrics such as precision, recall, and F1-score, along with visual confusion matrices.

---

## 🧩 Workflow Overview

For each trained model directory (`th_0.5_run_1`,`th_0.5_run_2`,.....,`th_0.55_run_1`,.....`th_0.95_run_1`, `th_0.95_run_2`, etc.), the following steps are performed:

1. **Model Loading**  
   Load the final checkpoint (`fasterrcnn_selftrained_final.pth`) from each training run.

2. **Prediction on Validation Data**  
   Use the model to generate object detection predictions on all validation images.

3. **IoU-Based Filtering**  
   For each ground-truth bounding box in the validation set, retain only the best-matching predicted box with an IoU > 0.5 and the highest confidence score.

4. **Results Merging**  
   Aggregate all filtered predictions into a single CSV file.

5. **Performance Evaluation**  
   Compute the confusion matrix and classification metrics (precision, recall, F1-score) using the merged predictions compared against ground-truth labels.

6. **Output Generation**  
   - Save confusion matrix as a heatmap image
   - Save confusion matrix as raw CSV
   - Save precision, recall, and F1-score in a structured CSV file
   - Collect all run-wise results in a summary file

---

## 📁 Input Requirements

- `val_labels.csv`: CSV file containing ground-truth bounding boxes for validation images.
- `images/`: Folder containing `.tif` validation images referenced in `val_labels.csv`.
- `checkpoints/fasterrcnn_selftrained_final.pth`: Trained Faster R-CNN models from multiple runs.

---

## 📂 Output Files (Per Run)

Each run will generate the following artifacts:

- `predicted_boxes/*.csv` – Raw model predictions per image
- `filtered_predicted_dots/*.csv` – Filtered predictions using IoU and score
- `merged_predictions.csv` – Combined result for all validation images
- `confusion_matrix.png` – Heatmap visualization of the confusion matrix
- `confusion_matrix_raw.csv` – Raw matrix values in tabular form
- `classification_metrics.csv` – Precision, recall, and F1-score
- `all_checkpoints_metrics.csv` – Summary of metrics across all runs

---

## 📊 Evaluation Metrics

The script uses standard classification metrics based on predicted vs. true labels:

- **Precision (macro-averaged)**
- **Recall (macro-averaged)**
- **F1 Score (macro-averaged)**

All evaluations are performed across 5 tree species classes (labeled 0–4).

---

## 🔧 Dependencies

Required Python packages include:

- `torch`, `torchvision`
- `pandas`, `numpy`
- `Pillow`, `tqdm`
- `matplotlib`, `seaborn`
- `scikit-learn`

---

## 📈 Applications

This evaluation framework is suitable for:

- Benchmarking self-training pipelines
- Validating pseudo-labeled datasets
- Comparing models trained under different thresholds or settings

---

## 📄 License

MIT License
