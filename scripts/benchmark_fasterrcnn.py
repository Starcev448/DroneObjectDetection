import os
import time

import torch
import torchvision
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 11

IMAGE_DIR = "dataset/images/val"

transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])

model = fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features,
    NUM_CLASSES,
)

model.load_state_dict(torch.load("models/fasterrcnn_final.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

images = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.endswith(".jpg")
])

print("Изображений:", len(images))

times = []

with torch.no_grad():

    for img_path in images:

        image = Image.open(img_path).convert("RGB")
        image = transform(image).to(DEVICE)

        start = time.perf_counter()

        model([image])

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        times.append(end - start)

avg = sum(times) / len(times)

print(f"\nСреднее время: {avg:.4f} сек")
print(f"FPS: {1 / avg:.2f}")