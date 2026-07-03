"""
Общие классы датасетов для проекта.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from PIL import Image
from torchvision import transforms


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

NUM_CLASSES = len(CLASSES)

transform = transforms.ToTensor()


class VisDroneDataset(Dataset):

    def __init__(self, image_dir, label_dir, transforms=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.images = sorted(self.image_dir.glob("*.jpg"))
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        label_path = self.label_dir / (image_path.stem + ".txt")

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)
        else:
            image = transform(image)

        _, height, width = image.shape

        boxes = []
        labels = []

        if label_path.exists():

            with open(label_path, "r") as f:

                for line in f:

                    values = line.strip().split()

                    cls = int(values[0])

                    x = float(values[1])
                    y = float(values[2])
                    w = float(values[3])
                    h = float(values[4])

                    xmin = (x - w / 2) * width
                    ymin = (y - h / 2) * height
                    xmax = (x + w / 2) * width
                    ymax = (y + h / 2) * height

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls + 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": (
                (boxes[:, 2] - boxes[:, 0]) *
                (boxes[:, 3] - boxes[:, 1])
                if len(boxes) > 0
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        return image, target