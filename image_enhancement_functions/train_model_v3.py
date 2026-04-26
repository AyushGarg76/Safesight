import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ===== PATHS =====
# ===== PATHS =====
# Updated to point to the local dataset folder
data_dir = r"dataset"

# ===== TRANSFORMS =====
train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ===== DATA =====
# We load the dataset twice to apply different transforms to train/val
train_all = datasets.ImageFolder(data_dir, transform=train_transform)
val_all   = datasets.ImageFolder(data_dir, transform=val_transform)

# Generate indices and split
indices = list(range(len(train_all)))
np.random.shuffle(indices)
split = int(0.8 * len(indices))
train_indices, val_indices = indices[:split], indices[split:]

# Create subsets with correct transforms
train_data = torch.utils.data.Subset(train_all, train_indices)
val_data   = torch.utils.data.Subset(val_all,   val_indices)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Classes: {train_all.classes}")

# ===== MODEL =====
# FIX 4: MobileNetV3-Large — better accuracy-per-parameter than V2
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

# FIX 5: Unfreeze more layers — last 6 blocks give more fine-tuning power
for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-6:].parameters():
    param.requires_grad = True

# FIX 6: Add Dropout before the final classifier to reduce overfitting
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, 2)
)
model = model.to(device)

# ===== LOSS =====
# FIX 7: Label smoothing — prevents the model from becoming overconfident
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# FIX 8: Differential learning rates
#   Backbone gets a 10× smaller LR than the new classifier head
optimizer = torch.optim.AdamW([
    {"params": model.features[-6:].parameters(), "lr": 3e-5},
    {"params": model.classifier.parameters(),    "lr": 3e-4},
], weight_decay=1e-4)

# FIX 9: Cosine annealing — smoothly decays LR instead of fixed rate
epochs    = 40
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# ===== TRACKING =====
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
best_acc     = 0
patience     = 7        # FIX 10: Early stopping
patience_ctr = 0

# ===== TRAIN LOOP =====
for epoch in range(epochs):
    # --- Train ---
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc  = correct / total

    # --- Validate ---
    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc   = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:02d} | Train {train_acc:.3f} | Val {val_acc:.3f} "
          f"| Loss {val_loss:.4f} | LR {current_lr:.2e}")

    # FIX 11: Save best checkpoint + detailed metrics
    if val_acc > best_acc:
        best_acc     = val_acc
        patience_ctr = 0
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names":      train_all.classes,
            "val_acc":          val_acc,
            "epoch":            epoch + 1,
        }, "helmet_model_best_v3.pth")
        print(f"  ✅ Saved best model (val_acc={best_acc:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            print(f"⚠️  Early stopping at epoch {epoch+1}")
            break

print(f"\n✅ Training done — Best val accuracy: {best_acc:.4f}")

# ===== FINAL EVALUATION =====
# FIX 12: Report precision / recall / F1 instead of accuracy alone
print("\n--- Classification report (best checkpoint) ---")
print(classification_report(all_labels, all_preds, target_names=train_all.classes))

# ===== PLOTS =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_accs, label="Train")
axes[0].plot(val_accs,   label="Val")
axes[0].set_title("Accuracy")
axes[0].legend()
axes[1].plot(train_losses, label="Train")
axes[1].plot(val_losses,   label="Val")
axes[1].set_title("Loss")
axes[1].legend()
plt.tight_layout()
plt.savefig("training_curves_v3.png", dpi=150)
plt.show()
