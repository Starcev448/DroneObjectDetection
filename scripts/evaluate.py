import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from datasets import VisDroneDataset


# ==========================
# НАСТРОЙКИ
# ==========================

MODEL_NAME = "fcos"
MODEL_PATH = "models/fcos_final.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 11

ROOT = "dataset"

VAL_IMAGES = os.path.join(ROOT, "images", "val")
VAL_LABELS = os.path.join(ROOT, "labels", "val")


transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])


dataset = VisDroneDataset(
    VAL_IMAGES,
    VAL_LABELS,
    transforms=transform,
)


def collate_fn(batch):
    return tuple(zip(*batch))


loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
)

# ==========================
# ЗАГРУЗКА МОДЕЛИ
# ==========================

if MODEL_NAME == "fasterrcnn":

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            NUM_CLASSES,
        )
    )

elif MODEL_NAME == "retinanet":

    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None)

    in_features = model.head.classification_head.cls_logits.in_channels

    num_anchors = model.head.classification_head.num_anchors

    model.head.classification_head = (
        torchvision.models.detection.retinanet.RetinaNetClassificationHead(
            in_features,
            num_anchors,
            NUM_CLASSES,
        )
    )

elif MODEL_NAME == "ssd":

    model = torchvision.models.detection.ssd300_vgg16(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES,
    )

elif MODEL_NAME == "fcos":

    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES,
    )

else:
    raise ValueError("Неизвестная модель")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.to(DEVICE)

model.eval()

metric = MeanAveragePrecision()


# ==========================
# ОЦЕНКА
# ==========================

with torch.no_grad():

    for images, targets in loader:

        images = [img.to(DEVICE) for img in images]

        outputs = model(images)

        outputs = [
            {
                "boxes": o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                "labels": o["labels"].cpu(),
            }
            for o in outputs
        ]

        targets = [
            {
                "boxes": t["boxes"],
                "labels": t["labels"],
            }
            for t in targets
        ]

        metric.update(outputs, targets)

result = metric.compute()

print("\n===========================")
print(MODEL_NAME.upper())
print("===========================")

print("mAP50      :", float(result["map_50"]))
print("mAP50-95   :", float(result["map"]))
print("Precision  :", float(result["map_50"]))
print("Recall     :", float(result["mar_100"]))