"""
Обучение YOLO11 на датасете VisDrone.

Автор: Практическая работа
"""

from ultralytics import YOLO


def main():

    # Загружаем предобученную модель
    model = YOLO("yolo11n.pt")

    # Обучение
    model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project="results",
        name="yolo11",
        exist_ok=True,
        pretrained=True,
        patience=15,
        verbose=True
    )


if __name__ == "__main__":
    main()