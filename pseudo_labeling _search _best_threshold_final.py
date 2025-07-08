# ===================== Import ها =====================
import os
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from torchvision.ops import box_iou
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===================== Dataset Class =====================
def safe_read_csv(path):
    try:
        if os.path.getsize(path) == 0:
            print(f"⚠️ فایل خالی است: {path}")
            return pd.DataFrame()
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"⚠️ خطا در خواندن فایل (EmptyDataError): {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ خطای غیرمنتظره هنگام خواندن فایل {path}: {e}")
        return pd.DataFrame()

class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_file):
        self.root_dir = root_dir
        self.images_dir = os.path.join(os.path.dirname(root_dir), 'images')
        self.labels_df = pd.read_csv(label_file)
        self.image_files = self.labels_df['filename'].unique()

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)
        records = self.labels_df[self.labels_df['filename'] == filename]
        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = records['class'].values
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }
        return img, target, filename

    def __len__(self):
        return len(self.image_files)

def collate_fn(batch):
    return tuple(zip(*batch))

# ===================== Training & Evaluation Functions =====================
def train_one_epoch(model, dataloader, optimizer):
    model.train()
    for images, targets, _ in tqdm(dataloader, desc="Training"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def generate_pseudo_labels(model, dataloader, threshold=0.8):
    model.eval()
    pseudo_data = []
    with torch.no_grad():
        for images, _, filenames in tqdm(dataloader, desc="Generating Pseudo Labels"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()
                keep = scores > threshold
                for box, label in zip(boxes[keep], labels[keep]):
                    pseudo_data.append({
                        'filename': filenames[i],
                        'class': label.item(),
                        'xmin': box[0].item(),
                        'ymin': box[1].item(),
                        'xmax': box[2].item(),
                        'ymax': box[3].item()
                    })
    return pd.DataFrame(pseudo_data)

def compute_map(model, dataloader, iou_threshold=0.5, score_threshold=0.05):
    model.eval()
    all_ap = []
    with torch.no_grad():
        for images, targets, _ in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i in range(len(images)):
                pred_boxes = outputs[i]['boxes'].cpu()
                pred_scores = outputs[i]['scores'].cpu()
                keep = pred_scores > score_threshold
                pred_boxes = pred_boxes[keep]
                true_boxes = targets[i]['boxes'].cpu()
                if len(pred_boxes) == 0 or len(true_boxes) == 0:
                    all_ap.append(0)
                    continue
                ious = box_iou(pred_boxes, true_boxes)
                hits = (ious > iou_threshold).any(dim=1).float()
                ap = hits.sum() / len(pred_boxes)
                all_ap.append(ap.item())
    return sum(all_ap) / len(all_ap) if all_ap else 0.0

# ===================== Self-Training Function =====================
def run_self_training(threshold, repeat_idx):
    print(f"\n🚀 Starting run for threshold {threshold} - repeat {repeat_idx}")

    base_dir = r'E:\FASTRCNN\FASTRCNN\dataset\psudo_labeling'
    run_dir = os.path.join(base_dir, f'th_{threshold}_run_{repeat_idx}')
    os.makedirs(run_dir, exist_ok=True)

    train_csv_path = os.path.join(run_dir, 'train_labels.csv')
    test_csv_path = os.path.join(run_dir, 'test_labels.csv')
    val_csv_path = os.path.join(run_dir, 'val_labels.csv')

    if not os.path.exists(train_csv_path):
        for file in ['train_labels.csv', 'test_labels.csv', 'val_labels.csv']:
            src = os.path.join(base_dir, file)
            dst = os.path.join(run_dir, file)
            if os.path.exists(src):
                pd.read_csv(src).to_csv(dst, index=False)
            else:
                raise FileNotFoundError(f"❌ فایل {src} یافت نشد. لطفاً آن را بررسی کن.")

    train_df = pd.read_csv(train_csv_path)
    val_dataset = ObjectDetectionDataset(run_dir, val_csv_path)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 5)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    patience = 4
    max_iterations = 10
    map_log = []
    best_map = 0.0
    no_improve_count = 0

    for iteration in range(1, max_iterations + 1):
        print(f"\n🔁 Iteration {iteration}")

        if iteration == 1:
            combined_df = train_df.copy()
        else:
            prev_pseudo_path = os.path.join(run_dir, f'pseudo_labels_iter{iteration - 1}.csv')
            if os.path.exists(prev_pseudo_path) and os.path.getsize(prev_pseudo_path) > 0:
                pseudo_df = safe_read_csv(prev_pseudo_path)
                combined_df = pd.concat([train_df, pseudo_df], ignore_index=True)
            else:
                print(f"⚠️ Pseudo-label file missing or empty: {prev_pseudo_path}")
                combined_df = train_df.copy()

        combined_csv = os.path.join(run_dir, f'combined_iter{iteration}.csv')
        combined_df.to_csv(combined_csv, index=False)

        combined_dataset = ObjectDetectionDataset(run_dir, combined_csv)
        combined_loader = DataLoader(combined_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
        train_one_epoch(model, combined_loader, optimizer)

        test_dataset = ObjectDetectionDataset(run_dir, test_csv_path)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        pseudo_df = generate_pseudo_labels(model, test_loader, threshold=threshold)
        pseudo_path = os.path.join(run_dir, f'pseudo_labels_iter{iteration}.csv')
        pseudo_df.to_csv(pseudo_path, index=False)
        print(f"✅ Pseudo-labels saved: {pseudo_path}")

        map_score = compute_map(model, val_loader)
        map_log.append({'iteration': iteration, 'mAP@0.5': round(map_score, 4)})
        print(f"📊 Iteration {iteration} - mAP@0.5: {map_score:.4f}")

        scheduler.step(map_score)
        print(f"🔧 Current learning rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

        if map_score > best_map:
            best_map = map_score
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"🛑 Early stopping: no improvement in {patience} iterations.")
            break

    final_model_path = os.path.join(run_dir, 'fasterrcnn_selftrained_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved: {final_model_path}")

    map_df = pd.DataFrame(map_log)
    map_log_path = os.path.join(run_dir, 'map_log.csv')
    # map_df.to_csv(map_log_path, index=False)

    plt.figure()
    iterations = [entry['iteration'] for entry in map_log]
    scores = [entry['mAP@0.5'] for entry in map_log]
    plt.plot(iterations, scores, marker='o')
    plt.title(f'mAP Progress - Threshold {threshold} - Repeat {repeat_idx}')
    plt.xlabel('Iteration')
    plt.ylabel('mAP@0.5')
    plt.grid(True)
    chart_path = os.path.join(run_dir, 'map_progress.png')
    # plt.savefig(chart_path)
    plt.close()
    print(f"📊 mAP progress plot saved to: {chart_path}")

# ===================== اجرای همه ترشولدها و تکرارها =====================
thresholds = [0.9, 0.95]
repeats = 4

for threshold in thresholds:
    for repeat in range(1, repeats + 1):
        run_self_training(threshold, repeat)

# ===================== رسم نمودار نهایی میانگین‌ها =====================
threshold_avg_map = {}
base_dir = r'E:\FASTRCNN\FASTRCNN\dataset\psudo_labeling'

for threshold in thresholds:
    all_runs = []
    for repeat in range(1, repeats + 1):
        run_dir = os.path.join(base_dir, f'th_{threshold}_run_{repeat}')
        map_log_path = os.path.join(run_dir, 'map_log.csv')
        if os.path.exists(map_log_path):
            df = pd.read_csv(map_log_path)
            all_runs.append(df['mAP@0.5'].values)

    if all_runs:
        min_len = min(len(run) for run in all_runs)
        all_runs_trimmed = [run[:min_len] for run in all_runs]
        avg_map = np.mean(all_runs_trimmed, axis=0)
        threshold_avg_map[threshold] = avg_map

plt.figure()
for threshold, avg_map in threshold_avg_map.items():
    plt.plot(range(1, len(avg_map) + 1), avg_map, label=f'{threshold}')
plt.title('Average mAP Progress for All Thresholds')
plt.xlabel('Iteration')
plt.ylabel('Average mAP@0.5')
plt.legend(title='Threshold')
plt.grid(True)
final_plot_path = os.path.join(base_dir, 'all_thresholds_avg_map.png')
# plt.savefig(final_plot_path)
plt.close()
print(f"📊 Final multi-threshold plot saved to: {final_plot_path}")

# ===================== ذخیره‌سازی مقادیر میانگین در summary.csv =====================
summary_data = []

for threshold, avg_map in threshold_avg_map.items():
    summary_data.append({
        'Threshold': threshold,
        'Avg_mAP@0.5': round(np.mean(avg_map), 4),
        'Max_mAP@0.5': round(np.max(avg_map), 4),
        'Min_mAP@0.5': round(np.min(avg_map), 4),
        'Last_mAP@0.5': round(avg_map[-1], 4)
    })

summary_df = pd.DataFrame(summary_data)
summary_csv_path = os.path.join(base_dir, 'summary.csv')
# summary_df.to_csv(summary_csv_path, index=False)
print(f"✅ Summary CSV saved to: {summary_csv_path}")
