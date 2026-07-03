import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

import cv2
import torch

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import functional as F

from detector_utils import (
    create_counter,
    print_statistics,
    save_statistics,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "__background__",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

MODEL_PATH = "models/retinanet_final.pth"
IMAGE_DIR = "dataset/images/val"
OUTPUT_DIR = "results/retinanet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Модель
# -----------------------------
model = retinanet_resnet50_fpn(weights=None)

num_anchors = model.head.classification_head.num_anchors
in_channels = model.backbone.out_channels

model.head.classification_head = RetinaNetClassificationHead(
    in_channels=in_channels,
    num_anchors=num_anchors,
    num_classes=len(CLASSES),
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )
)

model.to(DEVICE)
model.eval()

counter = create_counter()

# -----------------------------
# Предсказание
# -----------------------------
files = sorted(os.listdir(IMAGE_DIR))

for filename in files:

    if not filename.lower().endswith(".jpg"):
        continue

    path = os.path.join(IMAGE_DIR, filename)

    image = cv2.imread(path)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = F.to_tensor(rgb).to(DEVICE)

    with torch.no_grad():
        prediction = model([tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):

        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)

        class_name = CLASSES[label]
        counter[class_name] += 1

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"{class_name} {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, filename),
        image,
    )

print_statistics(counter)

save_statistics(
    counter,
    OUTPUT_DIR,
)

print("\nГотово!")