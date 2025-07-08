import os
import re
import pandas as pd
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from confusion_matrix import compute_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# تنظیمات دستگاه
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\U0001F680 Device:", device)

# مسیر اصلی داده‌ها
base_data_dir = r"E:\\FASTRCNN\\FASTRCNN\\dataset\\psudo_labeling"

# لیست پوشه‌هایی که می‌خواهیم مدل روی آن‌ها تست شود


run_folders = ["th_0.95_run_1", "th_0.95_run_2", "th_0.95_run_3", "th_0.95_run_4"]

# کلاس دیتاست
class trDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase):
        self.root = root
        self.phase = phase
        self.targets = pd.read_csv(os.path.join(root, f'{phase}_labels.csv'))
        self.imgs = self.targets['filename'].astype(str)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img)
        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

# تابع IoU

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    return inter_area / (area1 + area2 - inter_area + 1e-6)

# فیلتر فایل‌ها بر اساس IoU

def process_files(boxes_file, gt_df, output_file):
    try:
        boxes_df = pd.read_csv(boxes_file)
    except pd.errors.EmptyDataError:
        return
    filename = os.path.splitext(os.path.basename(boxes_file))[0] + '.tif'
    gt_boxes = gt_df[gt_df['filename'] == filename]
    output_rows = []
    for _, gt_row in gt_boxes.iterrows():
        gt_box = [gt_row['xmin'], gt_row['ymin'], gt_row['xmax'], gt_row['ymax']]
        best_box = None
        best_iou = -1
        best_score = -1
        for _, box_row in boxes_df.iterrows():
            pred_box = [box_row['xmin'], box_row['ymin'], box_row['xmax'], box_row['ymax']]
            score = box_row['score']
            iou = calculate_iou(gt_box, pred_box)
            if iou > 0.5 and (iou > best_iou or (iou == best_iou and score > best_score)):
                best_box = box_row
                best_iou = iou
                best_score = score
        if best_box is not None:
            output_rows.append(best_box.to_dict())
    if output_rows:
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(output_file, index=False)

# اجرای مدل‌ها
all_results = []

for run_name in run_folders:
    print(f"\n\U0001F680 Running on: {run_name}")
    run_path = os.path.join(base_data_dir, run_name)
    checkpoint_path = os.path.join(run_path, "checkpoints")
    model_path = os.path.join(checkpoint_path, "fasterrcnn_selftrained_final.pth")
    combine_result_dir = os.path.join(checkpoint_path, "COMBINE_RESULT")
    output_folder = os.path.join(combine_result_dir, "predicted_boxes")
    filtered_dots_folder = os.path.join(combine_result_dir, "filtered_predicted_dots")
    merged_file_path = os.path.join(combine_result_dir, "merged_predictions.csv")

    os.makedirs(combine_result_dir, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(filtered_dots_folder, exist_ok=True)

    # بارگذاری مدل
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 5)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # پیش‌بینی
    test_dataset = trDataset(base_data_dir, 'val')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="📸 Predicting"):
            images = [img.to(device) for img in images]
            out = model(images)
            for i, (filename, prediction) in enumerate(zip(filenames, out)):
                filename_str = os.path.splitext(os.path.basename(str(filename)))[0]
                output_csv_file = os.path.join(output_folder, f"{filename_str}.csv")
                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()
                data = [
                    [filename_str, label, int(xmin), int(ymin), int(xmax), int(ymax), float(score)]
                    for box, label, score in zip(boxes, labels, scores)
                    for xmin, ymin, xmax, ymax in [box]
                ]
                df = pd.DataFrame(data, columns=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])
                df.to_csv(output_csv_file, index=False)

    # فیلتر کردن
    gt_df = pd.read_csv(os.path.join(base_data_dir, "val_labels.csv"))
    for pred_filename in tqdm(os.listdir(output_folder), desc="🔍 Filtering IoU"):
        if pred_filename.endswith(".csv"):
            process_files(os.path.join(output_folder, pred_filename), gt_df, os.path.join(filtered_dots_folder, pred_filename))

    # ادغام خروجی‌ها
    merged_data = []
    for csv_file in os.listdir(filtered_dots_folder):
        file_path = os.path.join(filtered_dots_folder, csv_file)
        if os.stat(file_path).st_size == 0:
            continue
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            continue
        df['filename'] = df['filename'].astype(str).str.replace('.0', '', regex=False) + ".tif"
        df['score'] = pd.to_numeric(df['score'], errors='coerce') if 'score' in df.columns else 0.0
        merged_data.append(df)

    if merged_data:
        final_df = pd.concat(merged_data, ignore_index=True)
        final_df.to_csv(merged_file_path, index=False)

        # متریک‌ها
        conf_matrix, true_labels, pred_labels = compute_confusion_matrix(
            os.path.join(base_data_dir, "val_labels.csv"), merged_file_path
        )
        labels_present = sorted(np.unique(true_labels))
        precision = precision_score(true_labels, pred_labels, average='macro', labels=labels_present)
        recall = recall_score(true_labels, pred_labels, average='macro', labels=labels_present)
        f1 = f1_score(true_labels, pred_labels, average='macro', labels=labels_present)

        # ذخیره نتایج
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix ({run_name})")
        plt.savefig(os.path.join(combine_result_dir, "confusion_matrix.png"))
        plt.close()

        conf_matrix_df = pd.DataFrame(conf_matrix, index=[0, 1, 2, 3, 4], columns=[0, 1, 2, 3, 4])
        conf_matrix_df.to_csv(os.path.join(combine_result_dir, "confusion_matrix_raw.csv"), index_label="True/Pred")

        metrics_df = pd.DataFrame({"Metric": ["Precision", "Recall", "F1 Score"],
                                   "Value": [precision, recall, f1]})
        metrics_df.to_csv(os.path.join(combine_result_dir, "classification_metrics.csv"), index=False)

        all_results.append({
            "Run": run_name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })
    else:
        print(f"⚠️ No valid predictions for: {run_name}")

# خروجی نهایی تمام اجراها
if all_results:
    summary_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(base_data_dir, "all_checkpoints_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("\n📊 Summary of All Runs:")
    print(summary_df.to_string(index=False))
else:
    print("⚠️ No results to summarize.")
