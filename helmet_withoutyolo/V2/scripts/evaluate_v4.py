import os
import glob
import torch
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

NUM_CLASSES = 4
CLASS_NAMES = ['__background__', 'helmet', 'head', 'person']
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Assuming the script runs from helmet_withoutyolo/V2/scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'final')  # wait, where is dataset?
# Let's fix this properly. 
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'helmet_withoutyolo', 'dataset', 'archive')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
ANNOT_DIR = os.path.join(DATA_DIR, 'annotations')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'helmet_withoutyolo', 'savedmodel', 'best_model_v4.pth')

class EvalDataset(Dataset):
    def __init__(self, image_paths, annot_paths):
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.transforms = A.Compose([
            A.Resize(256, 256),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annot_path = self.annot_paths[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        boxes = []
        labels = []
        try:
            tree = ET.parse(annot_path)
            root = tree.getroot()
            
            wt = int(root.find('size').find('width').text)
            ht = int(root.find('size').find('height').text)

            for obj in root.findall('object'):
                label_name = obj.find('name').text
                if label_name not in CLASS_NAMES: continue
                labels.append(CLASS_NAMES.index(label_name))
                bbox = obj.find('bndbox')
                xmin = max(0, min(float(bbox.find('xmin').text), wt))
                ymin = max(0, min(float(bbox.find('ymin').text), ht))
                xmax = max(0, min(float(bbox.find('xmax').text), wt))
                ymax = max(0, min(float(bbox.find('ymax').text), ht))
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                else:
                    labels.pop()
        except Exception as e:
            print(f"Error parsing {annot_path}: {e}")

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        
        if len(boxes) > 0:
            sample = self.transforms(image=image, bboxes=target['boxes'].tolist(), labels=labels)
            image = sample['image']
            target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)
        else:
            image = ToTensorV2()(image=image)['image']
            target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
            target['labels'] = torch.empty((0,), dtype=torch.int64)
            
        return image, target

    def __len__(self):
        return len(self.image_paths)

def collate_fn(batch):
    return tuple(zip(*batch))

def create_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model

def evaluate():
    print(f"Using Device: {DEVICE}")
    print(f"Loading data from {DATA_DIR}...")
    all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    all_annots = [os.path.join(ANNOT_DIR, os.path.basename(img).replace('.png', '.xml')) for img in all_images]
    
    valid_pairs = [(img, ant) for img, ant in zip(all_images, all_annots) if os.path.exists(ant)]
    if not valid_pairs:
        print("No valid image-annotation pairs found!")
        return
        
    all_images, all_annots = zip(*valid_pairs)
    
    # Use validation split (last 10%) just like in training script
    split = int(0.9 * len(all_images))
    val_images, val_annots = all_images[split:], all_annots[split:]
    
    print(f"Total Validation Images: {len(val_images)}")
    
    dataset = EvalDataset(val_images, val_annots)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    print(f"Loading model from {MODEL_PATH}...")
    model = create_model().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Model path does not exist: {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Metrics
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox', class_metrics=True)

    iou_thresh = 0.5
    conf_thresh = 0.5
    
    tp = {1: 0, 2: 0, 3: 0} # 1: helmet, 2: head, 3: person
    fp = {1: 0, 2: 0, 3: 0}
    fn = {1: 0, 2: 0, 3: 0}

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            
            metric_targets = []
            metric_preds = []
            
            for t, o in zip(targets, outputs):
                gt_boxes = t['boxes'].to(DEVICE)
                gt_labels = t['labels'].to(DEVICE)
                
                pred_boxes = o['boxes']
                pred_labels = o['labels']
                pred_scores = o['scores']
                
                # Filter low confidence
                keep = pred_scores > conf_thresh
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]
                
                metric_targets.append({'boxes': gt_boxes, 'labels': gt_labels})
                metric_preds.append({'boxes': pred_boxes, 'scores': pred_scores, 'labels': pred_labels})
                
                # Custom TP/FP/FN logic
                for class_idx in range(1, NUM_CLASSES):
                    gt_idx = gt_labels == class_idx
                    pd_idx = pred_labels == class_idx
                    
                    gt_b = gt_boxes[gt_idx]
                    pd_b = pred_boxes[pd_idx]
                    
                    if len(gt_b) == 0 and len(pd_b) == 0:
                        continue
                    if len(gt_b) == 0 and len(pd_b) > 0:
                        fp[class_idx] += len(pd_b)
                        continue
                    if len(pd_b) == 0 and len(gt_b) > 0:
                        fn[class_idx] += len(gt_b)
                        continue
                        
                    ious = box_iou(gt_b, pd_b)
                    
                    matched_gts = set()
                    matched_preds = set()
                    
                    for gt_i in range(len(gt_b)):
                        best_pred = -1
                        best_iou = -1
                        for pd_i in range(len(pd_b)):
                            if pd_i in matched_preds: continue
                            if ious[gt_i, pd_i] > best_iou:
                                best_iou = ious[gt_i, pd_i]
                                best_pred = pd_i
                                
                        if best_iou > iou_thresh:
                            matched_gts.add(gt_i)
                            matched_preds.add(best_pred)
                            tp[class_idx] += 1
                            
                    fn[class_idx] += (len(gt_b) - len(matched_gts))
                    fp[class_idx] += (len(pd_b) - len(matched_preds))

            metric.update(metric_preds, metric_targets)

    print("\n" + "="*50)
    print(" OPTION 1: COCO mAP Metrics (torchmetrics)")
    print("="*50)
    map_results = metric.compute()
    print(f"mAP (IoU=0.50:0.95): {map_results['map']:.4f}")
    print(f"mAP (IoU=0.50)     : {map_results['map_50']:.4f}")
    print(f"mAP (IoU=0.75)     : {map_results['map_75']:.4f}")
    
    if 'map_per_class' in map_results:
        print("\nClass-wise mAP:")
        for idx, map_val in enumerate(map_results['map_per_class']):
            if map_val.item() != -1:
                class_id = map_results['classes'][idx].item()
                print(f"  {CLASS_NAMES[class_id]:<8}: {map_val.item():.4f}")

    print("\n" + "="*50)
    print(" OPTION 2: Custom Precision, Recall, F1 (IoU=0.5, Conf=0.5)")
    print("="*50)
    print(f"{'Class':<10} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("-" * 65)
    
    for c_id in range(1, NUM_CLASSES):
        c_tp = tp[c_id]
        c_fp = fp[c_id]
        c_fn = fn[c_id]
        
        precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
        recall = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{CLASS_NAMES[c_id]:<10} | {c_tp:<4} | {c_fp:<4} | {c_fn:<4} | {precision:.4f}    | {recall:.4f}    | {f1:.4f}")

    matrix_data = {
        'TP (Correct)': [tp[1], tp[2], tp[3]],
        'FP (Ghost)': [fp[1], fp[2], fp[3]],
        'FN (Missed)': [fn[1], fn[2], fn[3]]
    }
    df = pd.DataFrame(matrix_data, index=['helmet', 'head', 'person'])
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("Object Detection Confusion Matrix (IoU=0.5, Conf=0.5)")
    plt.ylabel("Object Class")
    plt.xlabel("Metric Type")
    plt.tight_layout()
    out_img = os.path.join('..', '..', 'eval_confusion_matrix.png')
    plt.savefig(out_img)
    print(f"\nSaved confusion matrix plot to: {os.path.abspath(out_img)}")

if __name__ == '__main__':
    evaluate()
