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

## 📌 Use Case

This Excel-based reporting structure enables:
- Transparent comparison between different confidence thresholds
- Easy identification of the optimal threshold value
- Reproducible analysis by copying new confusion matrix results into the predefined structure

---

## 📄 Notes

- All calculations are formula-based and automatically updated.
- Visual inspection of tables and PR metrics allows for a balanced decision in model selection.

---

## ✅ Conclusion

The Excel file provides a structured and automated way to analyze the performance of self-trained object detection models. Based on the current data, **threshold 0.7** offers the most balanced and accurate results among all tested configurations.
