import os
import time

from ultralytics import YOLO

IMAGE_DIR = "dataset/images/val"

model = YOLO("runs/detect/results/yolo11/weights/best.pt")

images = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.endswith(".jpg")
])

print("Изображений:", len(images))

times = []

for img in images:
    start = time.perf_counter()

    model.predict(
        img,
        verbose=False,
        save=False
    )

    end = time.perf_counter()

    times.append(end - start)

avg = sum(times) / len(times)

print(f"\nСреднее время: {avg:.4f} сек")
print(f"FPS: {1/avg:.2f}")