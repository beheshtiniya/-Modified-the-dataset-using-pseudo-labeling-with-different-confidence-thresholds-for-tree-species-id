import pandas as pd
import torch
from torchvision.ops import box_iou
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 🔧 فایل‌هایت را اینجا معرفی کن
gt_path = r"E:\FASTRCNN\FASTRCNN\dataset\psudo_labeling\old result\psuedo_th_0.8_lr_0.001 -2\val_labels.csv"
pred_path = r"E:\FASTRCNN\FASTRCNN\dataset\psudo_labeling\th_0.6_run_3\checkpoints\COMBINE_RESULT\merged_predictions.csv"

# Load data
gt_df = pd.read_csv(gt_path)
pred_df = pd.read_csv(pred_path)

iou_threshold = 0.5
all_classes = set(gt_df["class"]).union(set(pred_df["class"]))
aps = []
pr_curves = {}

# Group by image
gt_grouped = gt_df.groupby("filename")

for cls in sorted(all_classes):
    true_positives = []
    scores = []
    total_gt_boxes = 0
    gt_boxes_per_image = defaultdict(list)

    for filename, group in gt_grouped:
        boxes = group[group["class"] == cls][["xmin", "ymin", "xmax", "ymax"]].values
        if len(boxes) > 0:
            gt_boxes_per_image[filename] = {"boxes": torch.tensor(boxes, dtype=torch.float32), "detected": [False] * len(boxes)}
            total_gt_boxes += len(boxes)

    pred_data = pred_df[pred_df["class"] == cls].sort_values(by="score", ascending=False)

    for _, row in pred_data.iterrows():
        filename = row["filename"]
        pred_box = torch.tensor([[row["xmin"], row["ymin"], row["xmax"], row["ymax"]]], dtype=torch.float32)
        score = row["score"]
        scores.append(score)

        if filename in gt_boxes_per_image:
            gt_info = gt_boxes_per_image[filename]
            ious = box_iou(pred_box, gt_info["boxes"])[0]
            max_iou, max_idx = torch.max(ious, dim=0)
            if max_iou >= iou_threshold and not gt_info["detected"][max_idx]:
                true_positives.append(1)
                gt_info["detected"][max_idx] = True
            else:
                true_positives.append(0)
        else:
            true_positives.append(0)

    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum([1 - x for x in true_positives])
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (total_gt_boxes + 1e-6)

    pr_curves[cls] = (recalls, precisions)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precisions_at_recall = precisions[recalls >= t]
        p = max(precisions_at_recall) if len(precisions_at_recall) > 0 else 0
        ap += p / 11.0

    aps.append(ap)

map_50 = np.mean(aps)
print(f"\n✅ mAP@0.5 = {map_50:.4f}")

# 📈 رسم نمودار Precision-Recall
plt.figure(figsize=(8, 6))
for cls, (recalls, precisions) in pr_curves.items():
    plt.plot(recalls, precisions, label=f"Class {cls}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (per class)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
