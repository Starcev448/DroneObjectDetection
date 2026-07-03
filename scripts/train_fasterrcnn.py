import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torchvision

from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import v2 as T

from datasets import VisDroneDataset
from references.detection.engine import train_one_epoch, evaluate


# -----------------------------
# Настройки
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 11  # 10 классов + фон
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.005

ROOT = "dataset"
TRAIN_IMAGES = os.path.join(ROOT, "images", "train")
TRAIN_LABELS = os.path.join(ROOT, "labels", "train")


# -----------------------------
# Трансформации
# -----------------------------
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])


# -----------------------------
# Датасет
# -----------------------------
dataset = VisDroneDataset(
    TRAIN_IMAGES,
    TRAIN_LABELS,
    transforms=transform
)

print("Изображений:", len(dataset))


train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


def collate_fn(batch):
    return tuple(zip(*batch))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
)
# -----------------------------
# Модель
# -----------------------------
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    NUM_CLASSES,
)

model.to(DEVICE)


# -----------------------------
# Оптимизатор
# -----------------------------
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(
    params,
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=0.0005,
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1,
)


# -----------------------------
# Обучение
# -----------------------------
os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):
    print(f"\n========== ЭПОХА {epoch + 1}/{EPOCHS} ==========\n")

    train_one_epoch(
        model,
        optimizer,
        train_loader,
        DEVICE,
        epoch,
        print_freq=20,
    )

    lr_scheduler.step()

    torch.save(
        model.state_dict(),
        f"models/fasterrcnn_epoch_{epoch + 1}.pth",
)

torch.save(
    model.state_dict(),
    "models/fasterrcnn_final.pth",
)

print("\nОбучение завершено.")