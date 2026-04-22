import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# LOAD MODEL
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)

checkpoint = torch.load("helmet_model_best.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# LOAD VALIDATION DATA
val_data = datasets.ImageFolder(
    r"D:\Dev\Coding\Safesight\helmet_withoutyolo\V2\data\final\val",
    transform=transform
)

val_loader = DataLoader(val_data, batch_size=32)

# EVALUATE
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print("Validation Accuracy:", accuracy)