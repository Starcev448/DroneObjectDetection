import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

import cv2
import torch
import torchvision
from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from detector_utils import (
    create_counter,
    print_statistics,
    save_statistics,
)

# -----------------------------
# Настройки
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 11

MODEL_PATH = "models/fasterrcnn_final.pth"

INPUT_FOLDER = "dataset/images/val"
OUTPUT_FOLDER = "results/fasterrcnn"

SCORE_THRESHOLD = 0.5

CLASSES = [
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

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Модель
# -----------------------------
model = fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])

counter = create_counter()

# -----------------------------
# Предсказание
# -----------------------------
files = sorted(os.listdir(INPUT_FOLDER))

for filename in files:

    if not filename.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(INPUT_FOLDER, filename)

    image = cv2.imread(image_path)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = transform(rgb).to(DEVICE)

    with torch.no_grad():
        prediction = model([tensor])[0]

    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):

        if score < SCORE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)

        class_name = CLASSES[label - 1]

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
        os.path.join(OUTPUT_FOLDER, filename),
        image,
    )

print_statistics(counter)

save_statistics(
    counter,
    OUTPUT_FOLDER,
)

print("\nГотово!")