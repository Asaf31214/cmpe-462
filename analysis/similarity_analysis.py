#!/usr/bin/env python3
"""Compute intra- and inter-class similarities and detect outliers.

Saves:
- report/1d_similarity.txt   : Human-readable summary for the report
- report/outliers.csv        : Listed outliers (fused features)
- report/figures/plots/*     : Plots for distributions and centroids

This script reads processed tabular CSVs in `data/tabular/` and the
`vocabulary.json` to identify text columns. It computes cosine similarity
for image, text, numeric, and fused feature groups.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import polars as pl
except Exception:
    pl = None

from pathlib import Path


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load CSV and return (X, y, columns).
    Uses polars if available, otherwise numpy.genfromtxt fallback.
    """
    if pl is not None:
        df = pl.read_csv(path)
        cols = df.columns
        arr = df.to_numpy()
        X = arr[:, :-1].astype(float)
        y = arr[:, -1]
        return X, y, cols[:-1]

    # fallback
    import csv

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    arr = np.array(rows)
    X = arr[:, :-1].astype(float)
    y = arr[:, -1]
    return X, y, header[:-1]


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    # rows are samples
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    Xn = X / norms
    return Xn @ Xn.T


def centroid(X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((X.shape[1],), dtype=float)
    c = np.mean(X, axis=0)
    n = np.linalg.norm(c)
    if n == 0:
        return c
    return c / n


def stats_from_sims(sims: np.ndarray) -> Dict:
    sims = sims.flatten()
    sims = sims[~np.isnan(sims)]
    if sims.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "count": 0}
    return {"mean": float(np.mean(sims)), "std": float(np.std(sims)), "median": float(np.median(sims)), "count": int(sims.size)}


def ensure_dirs():
    Path("report/figures/plots").mkdir(parents=True, exist_ok=True)


def detect_outliers_to_centroid(X: np.ndarray, labels: np.ndarray, class_names: List[str]) -> List[Dict]:
    # return list of outlier dicts: {index, class, sim_to_centroid, zscore}
    outliers = []
    for cls in np.unique(labels):
        mask = labels == cls
        Xc = X[mask]
        if Xc.shape[0] < 2:
            continue
        cent = centroid(Xc)
        # cosine sim per sample to centroid
        sims = (Xc @ cent) / (np.linalg.norm(Xc, axis=1) * (np.linalg.norm(cent) + 1e-12))
        sims = np.nan_to_num(sims)
        mu = sims.mean()
        sigma = sims.std()
        if sigma == 0:
            continue
        z = (mu - sims) / sigma  # larger z means sample farther from centroid
        # flag z > 3 as outlier
        idxs = np.where(mask)[0]
        for i_local, zi in enumerate(z):
            if zi > 3:
                outliers.append({"idx": int(idxs[i_local]), "class": str(cls), "sim_to_centroid": float(sims[i_local]), "zscore": float(zi)})
    return outliers


def run():
    ensure_dirs()

    data_dir = Path("data/tabular")
    vocab_path = Path("data/vocabulary.json")

    # prefer fused processed; if not present fall back to images/text/combined
    fused_train = data_dir / "train_fusion_processed.csv"
    if not fused_train.exists():
        fused_train = data_dir / "train_processed.csv"
    train_path = fused_train

    X_train, y_train, cols = load_csv(str(train_path))

    # determine column groups
    vocab = []
    if vocab_path.exists():
        with open(vocab_path, "r") as f:
            vocab = json.load(f)

    cols = list(cols)
    image_cols = [c for c in cols if c.startswith("gray_") or c in {"blue_mean","blue_std","green_mean","green_std","red_mean","red_std"}]
    numeric_cols = [c for c in cols if c in {"weight","size"}]
    text_cols = [c for c in cols if c in vocab]

    # column indices
    col_idx = {c: i for i, c in enumerate(cols)}
    groups = {}
    if image_cols:
        groups["image"] = X_train[:, [col_idx[c] for c in image_cols]]
    if numeric_cols:
        groups["numeric"] = X_train[:, [col_idx[c] for c in numeric_cols]]
    if text_cols:
        groups["text"] = X_train[:, [col_idx[c] for c in text_cols]]

    groups["fused"] = X_train

    report_lines: List[str] = []
    report_lines.append("Intra- and inter-class similarity analysis (cosine similarity)\n")
    report_lines.append(f"Loaded {train_path} with {X_train.shape[0]} samples and {X_train.shape[1]} features.\n")

    for name, X in groups.items():
        report_lines.append(f"--- Feature group: {name} ---")
        # normalize features
        Xn = X.copy().astype(float)
        # handle zero-variance features
        norms = np.linalg.norm(Xn, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        # compute pairwise similarities per class (intra)
        intra_stats = {}
        intra_sims_all = []
        inter_sims_pairs = []

        for cls in np.unique(y_train):
            mask = y_train == cls
            Xc = Xn[mask]
            if Xc.shape[0] < 2:
                intra_stats[cls] = stats_from_sims(np.array([]))
                continue
            S = cosine_similarity_matrix(Xc)
            # take upper triangle without diagonal
            iu = np.triu_indices(S.shape[0], k=1)
            sims = S[iu]
            intra_stats[cls] = stats_from_sims(sims)
            intra_sims_all.append(sims)

        # inter: centroid similarities between classes
        class_centroids = {}
        classes = np.unique(y_train)
        for cls in classes:
            mask = y_train == cls
            Xc = Xn[mask]
            class_centroids[cls] = centroid(Xc)

        inter_matrix = np.zeros((len(classes), len(classes)), dtype=float)
        for i, a in enumerate(classes):
            for j, b in enumerate(classes):
                inter_matrix[i, j] = float(np.dot(class_centroids[a], class_centroids[b]))

        # sample-based inter similarity: between-class pairwise means (compute all pairs but avoid huge loops)
        sample_inter_means = []
        for i, a in enumerate(classes):
            mask_a = y_train == a
            for j, b in enumerate(classes):
                if j <= i:
                    continue
                mask_b = y_train == b
                Xa = Xn[mask_a]
                Xb = Xn[mask_b]
                if Xa.size == 0 or Xb.size == 0:
                    continue
                # compute pairwise via normalized dot
                Xa_n = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-12)
                Xb_n = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-12)
                sims = Xa_n @ Xb_n.T
                sample_inter_means.append(float(np.mean(sims)))

        report_lines.append("Per-class intra similarity stats:")
        for cls, s in intra_stats.items():
            report_lines.append(f"Class {cls}: mean={s['mean']:.4f}, std={s['std']:.4f}, median={s['median']:.4f}, pairs={s['count']}")

        report_lines.append("Centroid similarity matrix (rows/cols are classes):")
        report_lines.append(np.array2string(inter_matrix, precision=4, floatmode="fixed"))

        if sample_inter_means:
            report_lines.append(f"Sample-based inter-class mean similarity (mean across class-pairs) = {float(np.mean(sample_inter_means)):.4f}")

        # plots: histogram of intra (all classes merged) vs inter (sample-based)
        plt.figure(figsize=(8, 4))
        all_intra = np.concatenate(intra_sims_all) if intra_sims_all else np.array([])
        if all_intra.size:
            plt.hist(all_intra, bins=40, alpha=0.6, label="intra-class")
        if sample_inter_means:
            plt.hist(sample_inter_means, bins=40, alpha=0.6, label="inter-class (pair means)")
        plt.legend()
        plt.xlabel("Cosine similarity")
        plt.ylabel("Count")
        plt.title(f"Intra vs Inter similarity ({name})")
        plt.tight_layout()
        plt.savefig(f"report/figures/plots/{name}_intra_inter_hist.png")
        plt.close()

        # boxplot of per-class intra means
        class_means = [intra_stats[c]['mean'] for c in sorted(intra_stats.keys())]
        plt.figure(figsize=(8, 4))
        plt.boxplot(class_means, vert=False)
        plt.yticks([1], ["per-class intra mean"])
        plt.title(f"Per-class intra mean similarity ({name})")
        plt.tight_layout()
        plt.savefig(f"report/figures/plots/{name}_intra_box.png")
        plt.close()

    # Outlier detection on fused features
    outliers = detect_outliers_to_centroid(groups["fused"], y_train, list(np.unique(y_train)))
    if outliers:
        import csv

        out_path = Path("report/outliers.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "class", "sim_to_centroid", "zscore"])
            writer.writeheader()
            for o in outliers:
                writer.writerow(o)
        report_lines.append(f"Detected {len(outliers)} outliers (saved to {out_path}).")
    else:
        report_lines.append("No outliers detected (z>3) in fused features.")

    # write summary
    summary_path = Path("report/1d_similarity.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("Analysis complete. Summary written to", summary_path)


if __name__ == "__main__":
    run()
