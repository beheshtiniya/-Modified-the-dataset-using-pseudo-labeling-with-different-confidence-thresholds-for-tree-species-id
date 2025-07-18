# Self-Training Pipeline for Tree Species Detection using Faster R-CNN

This repository implements a semi-supervised object detection framework using **pseudo-labeling** and **Faster R-CNN** to improve tree species identification in aerial imagery. The system is designed to iteratively enhance model performance through self-training cycles based on varying confidence thresholds.

---

## 🎯 Objective

To train an object detection model on limited labeled data and iteratively improve it by leveraging its own predictions on unlabeled samples. The pipeline supports multiple training runs and evaluations at different confidence thresholds (e.g., 0.90, 0.95).

---

## 🔁 Method Overview

For each combination of threshold and training repeat:

1. **Initial Training**  
   The model is trained on a small manually labeled dataset (`train_labels.csv`).

2. **Pseudo-Label Generation**  
   The trained model predicts bounding boxes and labels on the test set. Predictions with confidence scores above the selected threshold are saved as pseudo-labels.

3. **Data Augmentation and Retraining**  
   Pseudo-labels are combined with the original training set to form a new training dataset. The model is retrained on this combined data.

4. **Iteration**  
   The process is repeated for multiple iterations to gradually refine the model.

5. **Validation**  
   After each iteration, the model is evaluated on a separate validation set (`val_labels.csv`) using the mAP@0.5 metric.

6. **Early Stopping**  
   Training stops early if validation performance does not improve over several iterations (default: 4).

---

## 📁 Input Structure

- `train_labels.csv`: Ground-truth annotations for training
- `test_labels.csv`: Unlabeled or weakly labeled data for pseudo-labeling
- `val_labels.csv`: Ground-truth for validation
- `images/`: Folder containing RGB images referenced in the CSVs

Each CSV must follow this format:
| filename | class | xmin | ymin | xmax | ymax |

---

## 📂 Output (Per Run)

Each training run (e.g., `th_0.90_run_1`) generates:

- `combined_iterN.csv`: Training data at iteration *N* (original + pseudo)
- `pseudo_labels_iterN.csv`: Pseudo-labels generated by the model
- `fasterrcnn_selftrained_final.pth`: Final trained model checkpoint

Additional logs, plots, and summaries can be optionally generated (some are disabled by default via comments in the script).

---

## 📊 Evaluation Metric

- **mAP@0.5** (mean Average Precision at IoU threshold 0.5)  
This metric is computed after each iteration to monitor improvement on the validation set.

---

## ⚙️ Configuration Parameters

- `thresholds`: List of confidence thresholds (e.g., [0.5, 0.55,..,0.90, 0.95])
- `repeats`: Number of times to repeat training for each threshold (e.g., 4)
- `max_iterations`: Maximum self-training iterations per run (default: 10)
- `patience`: Early stopping patience (default: 4)

---

## 🧠 Dependencies

Install required packages via:

```bash
pip install -r requirements.txt
