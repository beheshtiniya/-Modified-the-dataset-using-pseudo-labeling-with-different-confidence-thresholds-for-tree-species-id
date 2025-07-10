

## 📂 Execution Instructions (English)

1. **Download the images** from [yun.ir/q24k4b](https://yun.ir/q24k4b) and place them inside the `images/` folder.
2. Place the files `train_labels.csv`, `test_labels.csv`, and `val_labels.csv` from [yun.ir/q24k4b](https://yun.ir/q24k4b) in the same directory as the `images/` folder.  
3. Run the script `pseudo_labeling_search_best_threshold_final.py`.  
   ⚠️ **Before running**, update all file paths to match your local directory.  
   This script executes 4 self-training iterations for each threshold.  
4. Run `final_test.py` to generate predictions using the trained model, filter the predictions based on IoU, and compute the confusion matrix.  
5. Execute `map.py` to compute the final mAP metric.  
6. Copy the resulting confusion matrices into the appropriate section of the Excel file. Explanations are available in [`pseudo_labeling_results_explain.md`](pseudo_labeling_results_explain.md).
Metrics will be automatically calculated. Compare results across thresholds to identify the optimal one.

---
