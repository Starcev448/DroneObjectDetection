import os
import json
from collections import Counter


def save_statistics(counter, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    with open(
        os.path.join(save_dir, "statistics.json"),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dict(counter),
            f,
            indent=4,
            ensure_ascii=False,
        )


def print_statistics(counter):

    print("\n==============================")
    print("СТАТИСТИКА ПО КЛАССАМ")
    print("==============================")

    for name, count in sorted(counter.items()):
        print(f"{name:<20} {count}")

    print("==============================")
    print(f"Всего объектов: {sum(counter.values())}")


def create_counter():

    return Counter()