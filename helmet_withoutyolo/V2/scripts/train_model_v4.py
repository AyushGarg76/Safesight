import os
import torch
import torch.utils.data
import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Constants
NUM_CLASSES = 4 # ['__background__', 'helmet', 'head', 'person']
CLASS_NAMES = ['__background__', 'helmet', 'head', 'person']
BATCH_SIZE = 4 # Reduced for stability on most machines
NUM_EPOCHS = 10
NUM_WORKERS = 2
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Directories
DATA_DIR = os.path.join('..', '..', 'dataset', 'archive')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
ANNOT_DIR = os.path.join(DATA_DIR, 'annotations')
SAVE_DIR = os.path.join('..', '..', 'savedmodel')
os.makedirs(SAVE_DIR, exist_ok=True)

class SafetyHelmDataset(Dataset):
    def __init__(self, image_paths, annot_paths, class_names, transforms=None):
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.class_names = class_names
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annot_path = self.annot_paths[idx]

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # Parse XML
        boxes = []
        labels = []
        tree = ET.parse(annot_path)
        root = tree.getroot()
        
        wt = int(root.find('size').find('width').text)
        ht = int(root.find('size').find('height').text)

        for obj in root.findall('object'):
            label_name = obj.find('name').text
            if label_name not in self.class_names:
                continue
            
            labels.append(self.class_names.index(label_name))
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Clip boxes to image dimensions
            xmin = max(0, min(xmin, wt))
            ymin = max(0, min(ymin, ht))
            xmax = max(0, min(xmax, wt))
            ymax = max(0, min(ymax, ht))

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                # Remove label if box is invalid
                labels.pop()

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])

        if self.transforms:
            # Albumentations expects lists or numpy arrays, not torch tensors for labels/bboxes
            sample = self.transforms(image=image, bboxes=target['boxes'].tolist(), labels=labels.tolist())
            image = sample['image']
            target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)

        return image, target

    def __len__(self):
        return len(self.image_paths)

def get_train_transform():
    return A.Compose([
        A.Resize(128, 128),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transform():
    return A.Compose([
        A.Resize(128, 128),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def collate_fn(batch):
    return tuple(zip(*batch))

def create_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        return self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    train_loss_hist = Averager()
    prog_bar = tqdm(loader, total=len(loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        prog_bar.set_description(desc=f"Epoch {epoch+1} Train Loss: {loss_value:.4f}")

    return train_loss_hist.value

def validate(model, loader, device):
    model.train() # FasterRCNN needs train mode to compute loss
    val_loss_hist = Averager()
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss_hist.send(losses.item())

    return val_loss_hist.value

def main():
    print(f"Using device: {DEVICE}")

    # Load file lists
    all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    all_annots = [os.path.join(ANNOT_DIR, os.path.basename(img).replace('.png', '.xml')) for img in all_images]
    
    # Filter only those that exist
    valid_pairs = [(img, ant) for img, ant in zip(all_images, all_annots) if os.path.exists(ant)]
    all_images, all_annots = zip(*valid_pairs)
    
    # Split
    split = int(0.9 * len(all_images))
    train_images, val_images = all_images[:split], all_images[split:]
    train_annots, val_annots = all_annots[:split], all_annots[split:]

    train_dataset = SafetyHelmDataset(train_images, train_annots, CLASS_NAMES, get_train_transform())
    val_dataset = SafetyHelmDataset(val_images, val_annots, CLASS_NAMES, get_valid_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = create_model(NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        val_loss = validate(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(SAVE_DIR, 'best_model_v4.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Val Loss: {val_loss:.4f} to {save_path}")

if __name__ == '__main__':
    main()
