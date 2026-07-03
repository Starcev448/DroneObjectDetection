from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/results/yolo11/weights/best.pt")

    metrics = model.val(
        data="dataset.yaml",
        split="val",
        imgsz=640,
        conf=0.25,
        batch=8,
    )

    print("\n==============================")
    print("Результаты YOLO11")
    print("==============================")

    print(f"mAP50:       {metrics.box.map50:.4f}")
    print(f"mAP50-95:    {metrics.box.map:.4f}")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()