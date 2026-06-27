"""
Конвертация датасета VisDrone в формат COCO.

Автор: Практическая работа
Тема: Детекция объектов с дрона
"""

import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# -------------------------------------------------------
# Пути к данным
# -------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "dataset" / "VisDrone2019-DET-train"
VAL_DIR = PROJECT_DIR / "dataset" / "VisDrone2019-DET-val"

OUTPUT_DIR = PROJECT_DIR / "dataset" / "coco"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Классы VisDrone
# -------------------------------------------------------

CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]
# -------------------------------------------------------
# Конвертация одного набора данных
# -------------------------------------------------------


def convert_dataset(dataset_path: Path, output_json: Path):

    images = []
    annotations = []

    image_id = 1
    annotation_id = 1

    image_folder = dataset_path / "images"
    annotation_folder = dataset_path / "annotations"

    txt_files = sorted(annotation_folder.glob("*.txt"))

    print(f"\nОбработка: {dataset_path.name}")
    print(f"Найдено файлов: {len(txt_files)}")

    for txt_file in tqdm(txt_files):

        image_file = image_folder / (txt_file.stem + ".jpg")

        if not image_file.exists():
            continue

        width, height = Image.open(image_file).size

        images.append({
            "id": image_id,
            "file_name": image_file.name,
            "width": width,
            "height": height
        })

        with open(txt_file, "r") as f:

            for line in f:

                values = line.strip().split(",")

                if len(values) != 8:
                    continue

                x = float(values[0])
                y = float(values[1])
                w = float(values[2])
                h = float(values[3])

                score = int(values[4])
                category = int(values[5])

                if category == 0:
                    continue

                area = w * h

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1    
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print()
    print("Готово!")
    print(f"Изображений : {len(images)}")
    print(f"Объектов    : {len(annotations)}")
    print(f"Файл сохранён:")
    print(output_json)


# -------------------------------------------------------
# Запуск
# -------------------------------------------------------

if __name__ == "__main__":

    convert_dataset(
        TRAIN_DIR,
        OUTPUT_DIR / "instances_train.json"
    )

    convert_dataset(
        VAL_DIR,
        OUTPUT_DIR / "instances_val.json"
    )