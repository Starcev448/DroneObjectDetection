MODELS = [
    {
        "name": "YOLO11",
        "map50": 0.2192,
        "map5095": 0.1361,
        "precision": 0.5321,
        "recall": 0.2691,
        "fps": 68.73,
        "size": 5.3,
    },
    {
        "name": "Faster R-CNN",
        "map50": 0.3610,
        "map5095": 0.2098,
        "precision": 0.3610,
        "recall": 0.3023,
        "fps": 12.70,
        "size": 159,
    },
    {
        "name": "RetinaNet",
        "map50": 0.2279,
        "map5095": 0.1317,
        "precision": 0.2279,
        "recall": 0.2465,
        "fps": 12.90,
        "size": 124,
    },
    {
        "name": "SSD300",
        "map50": 0.0963,
        "map5095": 0.0468,
        "precision": 0.0963,
        "recall": 0.0799,
        "fps": 47.55,
        "size": 96,
    },
    {
        "name": "FCOS",
        "map50": 0.1974,
        "map5095": 0.1146,
        "precision": 0.1974,
        "recall": 0.2123,
        "fps": 14.04,
        "size": 123,
    },
]

print("=" * 102)
print(
    f'{"Модель":<18}'
    f'{"mAP50":>10}'
    f'{"mAP50-95":>12}'
    f'{"Precision":>12}'
    f'{"Recall":>10}'
    f'{"FPS":>10}'
    f'{"Размер(MB)":>15}'
)
print("=" * 102)

for m in MODELS:
    print(
        f'{m["name"]:<18}'
        f'{m["map50"]:>10.4f}'
        f'{m["map5095"]:>12.4f}'
        f'{m["precision"]:>12.4f}'
        f'{m["recall"]:>10.4f}'
        f'{m["fps"]:>10.2f}'
        f'{m["size"]:>15.1f}'
    )

print("=" * 102)

best_quality = max(MODELS, key=lambda x: x["map50"])
best_speed = max(MODELS, key=lambda x: x["fps"])
smallest = min(MODELS, key=lambda x: x["size"])

print()
print(f'Лучшая по качеству (mAP50): {best_quality["name"]}')
print(f'Самая быстрая (FPS): {best_speed["name"]}')
print(f'Самая компактная: {smallest["name"]}')