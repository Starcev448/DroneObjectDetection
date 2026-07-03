from ultralytics import YOLO

from detector_utils import (
    create_counter,
    print_statistics,
    save_statistics,
)


def main():

    model = YOLO("runs/detect/results/yolo11/weights/best.pt")

    results = model.predict(
        source="dataset/images/val",
        imgsz=640,
        conf=0.25,
        save=True,
        project="results",
        name="yolo11_predict",
        exist_ok=True,
    )

    counter = create_counter()

    for result in results:

        for cls in result.boxes.cls.tolist():

            counter[model.names[int(cls)]] += 1

    print_statistics(counter)

    save_statistics(
        counter,
        str(results[0].save_dir),
    )

    print("\nГотово!")


if __name__ == "__main__":
    main()