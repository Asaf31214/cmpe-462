import os
import csv
import random
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tabular")

def make_row(cls):
    # basic color means per class
    base = {
        "banana": (200, 180, 50),
        "carrot": (180, 100, 40),
        "cucumber": (60, 140, 60),
        "mandarina": (230, 110, 40),
        "tomato": (150, 40, 30),
    }
    b_mean, g_mean, r_mean = base.get(cls, (120, 120, 120))
    row = {}
    # color channel stats
    row["blue_mean"] = float(np.random.normal(b_mean, 20))
    row["blue_std"] = float(abs(np.random.normal(30, 5)))
    row["green_mean"] = float(np.random.normal(g_mean, 20))
    row["green_std"] = float(abs(np.random.normal(30, 5)))
    row["red_mean"] = float(np.random.normal(r_mean, 20))
    row["red_std"] = float(abs(np.random.normal(30, 5)))

    # 8x8 gray flattened
    gray = np.clip(np.random.normal((r_mean+g_mean+b_mean)/3, 40, size=(64,)), 0, 255)
    for i, val in enumerate(gray):
        row[f"gray_{i:03d}"] = float(val)

    # weight and size sample from class-specific distributions
    params = {
        "banana": {"weight": (120, 15), "size": (18, 2)},
        "carrot": {"weight": (60, 10), "size": (15, 2.5)},
        "cucumber": {"weight": (300, 40), "size": (20, 3)},
        "mandarina": {"weight": (80, 12), "size": (6.5, 0.8)},
        "tomato": {"weight": (100, 15), "size": (7, 1)},
    }
    p = params[cls]
    row["weight"] = float(np.random.normal(p["weight"][0], p["weight"][1]))
    row["size"] = float(np.random.normal(p["size"][0], p["size"][1]))
    row["class"] = cls
    return row

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    classes = ["banana", "carrot", "cucumber", "mandarina", "tomato"]
    samples_per_class = 600
    out_path = os.path.join(DATA_DIR, "feature_extraction.csv")
    print(f"Writing synthetic features to: {out_path}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = None
        for cls in classes:
            for _ in range(samples_per_class):
                row = make_row(cls)
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(row)

if __name__ == "__main__":
    main()
