import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import torch

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = retinanet_resnet50_fpn(weights=None)

num_anchors = model.head.classification_head.num_anchors
in_channels = model.backbone.out_channels

model.head.classification_head = RetinaNetClassificationHead(
    in_channels=in_channels,
    num_anchors=num_anchors,
    num_classes=11,
)

model.load_state_dict(
    torch.load(
        "models/retinanet_final.pth",
        map_location=DEVICE,
    )
)

model.to(DEVICE)
model.eval()

IMAGE_DIR = "dataset/images/val"

files = sorted(os.listdir(IMAGE_DIR))

print("Изображений:", len(files))

start = time.time()

for filename in files:

    image = cv2.imread(os.path.join(IMAGE_DIR, filename))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = F.to_tensor(rgb).to(DEVICE)

    with torch.no_grad():
        model([tensor])

end = time.time()

avg = (end - start) / len(files)

print()

print("Среднее время:", round(avg, 4), "сек")
print("FPS:", round(1 / avg, 2))