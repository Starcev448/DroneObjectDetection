import argparse
import subprocess
import sys

MODELS = {
    "yolo": "scripts/test_yolo.py",
    "fasterrcnn": "scripts/test_fasterrcnn.py",
    "retinanet": "scripts/test_retinanet.py",
    "ssd": "scripts/test_ssd.py",
    "fcos": "scripts/test_fcos.py",
}


def main():

    parser = argparse.ArgumentParser(
        description="Система анализа снимков с БПЛА"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS.keys(),
        help="Выберите модель для тестирования",
    )

    args = parser.parse_args()

    script = MODELS[args.model]

    print("=" * 60)
    print("Система анализа снимков с БПЛА")
    print("=" * 60)
    print(f"Используемая модель: {args.model.upper()}")
    print("=" * 60)

    subprocess.run(
        [sys.executable, script],
        check=True,
    )

    print("\nТестирование завершено.")


if __name__ == "__main__":
    main()