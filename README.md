# Threshold-Wise Evaluation Summary (Based on Excel Analysis)

The Excel file provided in this repository contains a detailed summary of evaluation metrics for different confidence thresholds used in pseudo-labeling during tree species detection.

---

## 📊 Sheet: `METRICS_AFTER_EVERY_THRESHOLD`

This sheet contains the **confusion matrix outputs** for all self-training iterations across multiple confidence thresholds (e.g., 0.6, 0.7, 0.8, 0.9, etc.).

To analyze the results:

1. The confusion matrix values from `METRICS_AFTER_EVERY_THRESHOLD` must be **copied into the `Calculating metrics1` sheet**.
2. Once pasted, the corresponding **evaluation metrics** (such as accuracy, precision, recall, and F1-score) are automatically computed in:
   - `Calculating metrics2`
   - `Calculating metrics3`

---

## 📈 Sheet-Level Summary

- `Calculating metrics2` and `Calculating metrics3`:  
  Automatically calculate class-wise and overall performance metrics for each threshold and iteration.

- Threshold-specific comparison tables below these sheets present:
  - Iteration-wise evaluation for each threshold
  - Side-by-side comparison of all thresholds across common metrics

---

## 🏆 Sheet: `The best results from the tables below`

This summary table highlights:
- The **best result of each iteration** for every threshold
- The **average performance across iterations**

From the aggregated comparison, it is observed that:

> **Threshold 0.7 consistently yields the best overall metrics across iterations**,  
> making it the most effective confidence threshold in this pseudo-labeling framework.

---

## 📂 Execution Instructions (English)

1. **Download the images** and place them inside the `images/` folder.  
2. Place the files `train_labels.csv`, `test_labels.csv`, and `val_labels.csv` in the same directory as the `images/` folder.  
3. Run the script `pseudo_labeling_search_best_threshold_final.py`.  
   ⚠️ **Before running**, update all file paths to match your local directory.  
   This script executes 4 self-training iterations for each threshold.  
4. Run `final_test.py` to generate predictions using the trained model, filter the predictions based on IoU, and compute the confusion matrix.  
5. Execute `map.py` to compute the final mAP metric.  
6. Copy the resulting confusion matrices into the appropriate section of the Excel file. Metrics will be automatically calculated. Compare results across thresholds to identify the optimal one.

---

## 📝 دستورالعمل اجرا (فارسی)

1. تصاویر را دانلود کرده و در پوشه `images/` قرار دهید. 
---

## ✅ Conclusion

The Excel file provides a structured and automated way to analyze the performance of self-trained object detection models. Based on the current data, **threshold 0.7** offers the most balanced and accurate results among all tested configurations.
