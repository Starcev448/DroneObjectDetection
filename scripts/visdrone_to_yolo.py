"""
Конвертация VisDrone в формат YOLO.

Структура после работы скрипта:

dataset/
│
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/
"""

from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "dataset" / "VisDrone2019-DET-train"
VAL_DIR = PROJECT_DIR / "dataset" / "VisDrone2019-DET-val"

OUTPUT_IMAGES = PROJECT_DIR / "dataset" / "images"
OUTPUT_LABELS = PROJECT_DIR / "dataset" / "labels"


def convert(split_name, source_dir):

    image_output = OUTPUT_IMAGES / split_name
    label_output = OUTPUT_LABELS / split_name

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    annotation_dir = source_dir / "annotations"
    image_dir = source_dir / "images"

    txt_files = sorted(annotation_dir.glob("*.txt"))

    print(f"\n=== {split_name.upper()} ===")
    print(f"Файлов: {len(txt_files)}")

    for txt in tqdm(txt_files):

        image_file = image_dir / (txt.stem + ".jpg")

        if not image_file.exists():
            continue

        shutil.copy2(
            image_file,
            image_output / image_file.name
        )

        width, height = Image.open(image_file).size

        output_label = label_output / txt.name

        with open(txt, "r") as fin, open(output_label, "w") as fout:

            for line in fin:

                values = line.strip().split(",")

                if len(values) != 8:
                    continue

                x = float(values[0])
                y = float(values[1])
                w = float(values[2])
                h = float(values[3])

                category = int(values[5])

                if category == 0 or category == 11:
                    continue

                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height

                w /= width
                h /= height

                class_id = category - 1

                fout.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}\n"
                )


if __name__ == "__main__":

    convert("train", TRAIN_DIR)
    convert("val", VAL_DIR)

    print("\nГотово!")