from ultralytics import YOLO


def main():
    model = YOLO("runs/detect/results/yolo11/weights/best.pt")

    model.predict(
        source="dataset/images/val",
        imgsz=640,
        conf=0.25,
        save=True,
        project="results",
        name="yolo11_predict",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()