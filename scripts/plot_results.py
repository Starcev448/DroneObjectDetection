import matplotlib.pyplot as plt

MODELS = [
    "YOLO11",
    "Faster R-CNN",
    "RetinaNet",
    "SSD300",
    "FCOS",
]

MAP50 = [
    0.2192,
    0.3610,
    0.2279,
    0.0963,
    0.1974,
]

FPS = [
    68.73,
    12.70,
    12.90,
    47.55,
    14.04,
]

SIZE = [
    5.3,
    159,
    124,
    96,
    123,
]

# -----------------------------
# График mAP50
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(MODELS, MAP50)
plt.title("Сравнение моделей по mAP50")
plt.ylabel("mAP50")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/map50.png", dpi=300)
plt.close()

# -----------------------------
# График FPS
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(MODELS, FPS)
plt.title("Сравнение моделей по скорости")
plt.ylabel("FPS")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/fps.png", dpi=300)
plt.close()

# -----------------------------
# Размер моделей
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(MODELS, SIZE)
plt.title("Размер файлов моделей")
plt.ylabel("МБ")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("results/model_size.png", dpi=300)
plt.close()

print("Графики успешно сохранены!")