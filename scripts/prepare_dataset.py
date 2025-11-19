import os
import sys
import random

import polars as pl


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_tabular = os.path.join(repo_root, "data", "tabular")
    src_paths = [
        os.path.join(data_tabular, "feature_extraction_augmented.csv"),
        os.path.join(data_tabular, "feature_extraction_filtered.csv"),
        os.path.join(data_tabular, "feature_extraction.csv"),
    ]

    input_csv = None
    for p in src_paths:
        if os.path.exists(p):
            input_csv = p
            break

    if input_csv is None:
        print("Could not find a feature CSV. Run `feature_extraction.ipynb` first to generate `data/tabular/feature_extraction.csv`.")
        sys.exit(1)

    print(f"Loading features from: {input_csv}")
    df = pl.read_csv(input_csv)

    # Add categorical attributes if not present
    if "color" not in df.columns:
        color_map = {
            "banana": "yellow",
            "carrot": "orange",
            "cucumber": "green",
            "mandarina": "orange",
            "tomato": "red",
        }
        classes_list = df["class"].to_list()
        color_list = [color_map.get(c, "unknown") for c in classes_list]
        df = df.with_columns(pl.Series(name="color", values=color_list))

    if "season" not in df.columns:
        seasons = ["spring", "summer", "autumn", "winter"]
        classes_list = df["class"].to_list()
        season_list = [random.choice(seasons) for _ in classes_list]
        df = df.with_columns(pl.Series(name="season", values=season_list))

    if "origin" not in df.columns:
        origins = ["Local", "Spain", "Turkey", "Netherlands", "Morocco"]
        classes_list = df["class"].to_list()
        origin_list = [random.choice(origins) for _ in classes_list]
        df = df.with_columns(pl.Series(name="origin", values=origin_list))

    # Add a one-sentence text description per sample
    if "text" not in df.columns:
        # build text descriptions from existing columns
        classes_list = df["class"].to_list()
        color_list = df["color"].to_list() if "color" in df.columns else [""] * len(classes_list)
        weight_list = df["weight"].to_list() if "weight" in df.columns else [None] * len(classes_list)
        size_list = df["size"].to_list() if "size" in df.columns else [None] * len(classes_list)

        texts = []
        for cls, color, weight, size in zip(classes_list, color_list, weight_list, size_list):
            parts = [f"A {color}" if color else "A fruit", cls]
            if weight is not None:
                parts.append(f"weighing {int(round(float(weight)))}g")
            if size is not None:
                parts.append(f"about {round(float(size),1)}cm")
            texts.append(" ".join(parts) + ".")

        df = df.with_columns(pl.Series(name="text", values=texts))

    # Ensure no missing values: fill numeric NAs with mean, categorical with mode
    for col in df.columns:
        if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
            if df[col].null_count() > 0:
                mean_val = float(df[col].mean())
                df = df.with_columns(pl.col(col).fill_null(mean_val))
        else:
            if df[col].null_count() > 0:
                # fill with "unknown"
                df = df.with_columns(pl.col(col).fill_null("unknown"))

    out_path = os.path.join(data_tabular, "feature_extraction_augmented.csv")
    print(f"Writing augmented features to: {out_path}")
    os.makedirs(data_tabular, exist_ok=True)
    df.write_csv(out_path)


if __name__ == "__main__":
    main()
