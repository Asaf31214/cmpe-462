"""
Train logistic regression implemented from scratch (one-vs-all) on the provided features.
Produces training/validation loss plots and compares performance/runtime with scikit-learn's LogisticRegression.

Usage:
  python scripts/logistic_scratch.py

Notes:
  - Expects `data/tabular/feature_extraction_augmented.csv` (created by `scripts/prepare_dataset.py`) or falls
    back to `feature_extraction_filtered.csv` or `feature_extraction.csv`.
"""
import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=200, reg=0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.w = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, X_val=None, y_val=None):
        n, d = X.shape
        self.w = np.zeros(d + 1)
        Xb = np.hstack([np.ones((n, 1)), X])
        losses = []
        val_losses = []
        for e in range(self.epochs):
            logits = Xb.dot(self.w)
            preds = self._sigmoid(logits)
            # cross-entropy loss
            eps = 1e-12
            loss = -np.mean(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
            # add L2 (skip bias)
            loss += 0.5 * self.reg * np.sum(self.w[1:] ** 2)
            losses.append(loss)

            # gradient
            grad = Xb.T.dot(preds - y) / n
            grad[1:] += self.reg * self.w[1:]
            self.w -= self.lr * grad

            if X_val is not None:
                Xbv = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
                pv = self._sigmoid(Xbv.dot(self.w))
                val_loss = -np.mean(y_val * np.log(pv + eps) + (1 - y_val) * np.log(1 - pv + eps))
                val_loss += 0.5 * self.reg * np.sum(self.w[1:] ** 2)
                val_losses.append(val_loss)

        return np.array(losses), np.array(val_losses) if len(val_losses) else None

    def predict_proba(self, X):
        Xb = np.hstack([np.ones((X.shape[0], 1)), X])
        return self._sigmoid(Xb.dot(self.w))

    def predict(self, X, thresh=0.5):
        return (self.predict_proba(X) >= thresh).astype(int)


def one_vs_all_train(X, y, lr=0.1, epochs=200, reg=0.0, val_fraction=0.2):
    classes = sorted(list(set(y)))
    models = {}
    history = {}
    for cls in classes:
        y_bin = (y == cls).astype(int)
        X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=val_fraction, random_state=462)
        model = LogisticRegressionScratch(lr=lr, epochs=epochs, reg=reg)
        losses, val_losses = model.fit(X_train, y_train, X_val, y_val)
        models[cls] = model
        history[cls] = (losses, val_losses)
    return models, history


def predict_ovo(models, X):
    classes = list(models.keys())
    probs = np.vstack([models[c].predict_proba(X) for c in classes]).T
    preds = np.array(classes)[np.argmax(probs, axis=1)]
    return preds, probs


def load_data():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_tabular = os.path.join(repo_root, "data", "tabular")
    candidates = [
        os.path.join(data_tabular, "feature_extraction_augmented.csv"),
        os.path.join(data_tabular, "feature_extraction_filtered.csv"),
        os.path.join(data_tabular, "feature_extraction.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"Loading {p}")
            df = pl.read_csv(p)
            return df
    raise FileNotFoundError("No feature CSV found. Run feature extraction and/or `scripts/prepare_dataset.py` first.")


def prepare_features(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict]:
    # identify image-derived columns (blue/green/red/gray)
    img_cols = [c for c in df.columns if c.startswith(("blue_", "green_", "red_", "gray_"))]
    num_cols = [c for c in ["weight", "size"] if c in df.columns]
    cat_cols = [c for c in ["color", "season", "origin"] if c in df.columns]
    text_col = "text" if "text" in df.columns else None

    X_parts = {}
    if img_cols:
        X_parts["image"] = df.select(img_cols).to_numpy()
    if num_cols:
        X_parts["numeric"] = df.select(num_cols).to_numpy()
    if cat_cols:
        cat_df = df.select(cat_cols).to_pandas()
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_onehot = enc.fit_transform(cat_df)
        X_parts["categorical"] = cat_onehot
    if text_col:
        texts = df[text_col].to_list()
        vec = TfidfVectorizer(max_features=100)
        X_text = vec.fit_transform(texts).toarray()
        X_parts["text"] = X_text

    y = df["class"].to_numpy()
    return X_parts, y, {"img_cols": img_cols, "num_cols": num_cols, "cat_cols": cat_cols, "text_col": text_col}


def run():
    df = load_data()
    X_parts, y, meta = prepare_features(df)

    # modality sets: image-only, numeric+categorical, text-only, fused
    modalities = {}
    if "image" in X_parts:
        modalities["image"] = X_parts["image"]
    if "numeric" in X_parts or "categorical" in X_parts:
        parts = []
        if "numeric" in X_parts:
            parts.append(X_parts["numeric"])
        if "categorical" in X_parts:
            parts.append(X_parts["categorical"])
        modalities["num_cat"] = np.hstack(parts)
    if "text" in X_parts:
        modalities["text"] = X_parts["text"]

    # fused
    fused_parts = [X_parts[k] for k in ("image", "numeric", "categorical", "text") if k in X_parts]
    if fused_parts:
        modalities["fused"] = np.hstack(fused_parts)

    results = {}
    for name, X in modalities.items():
        print(f"\nTraining modality: {name} with shape {X.shape}")
        # standardize
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=462, stratify=y)

        # create a validation set of 500 if enough samples, otherwise 20%
        if X_train.shape[0] > 600:
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=500, random_state=462, stratify=y_train)
        else:
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=462, stratify=y_train)

        t0 = time.time()
        models, history = one_vs_all_train(X_train_sub, y_train_sub, lr=0.5, epochs=300, reg=1e-3, val_fraction=0.2)
        t1 = time.time()
        train_time = t1 - t0

        preds, probs = predict_ovo(models, X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        }

        # AUC: compute one-vs-rest macro AUC if possible
        try:
            classes = sorted(list(set(y)))
            y_test_bin = np.vstack([(y_test == c).astype(int) for c in classes]).T
            aucs = []
            for i, c in enumerate(classes):
                aucs.append(roc_auc_score(y_test_bin[:, i], probs[:, i]))
            metrics["auc_macro"] = float(np.mean(aucs))
        except Exception:
            metrics["auc_macro"] = None

        # sklearn comparison
        t0 = time.time()
        sk = SklearnLR(max_iter=2000, multi_class="ovr")
        sk.fit(X_train_sub, y_train_sub)
        t1 = time.time()
        sk_time = t1 - t0
        sk_preds = sk.predict(X_test)
        sk_metrics = {
            "accuracy": accuracy_score(y_test, sk_preds),
            "precision_macro": precision_score(y_test, sk_preds, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, sk_preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, sk_preds, average="macro", zero_division=0),
        }

        results[name] = {
            "scratch": {"metrics": metrics, "train_time": train_time},
            "sklearn": {"metrics": sk_metrics, "train_time": sk_time},
            "history": history,
        }

        # plot training/validation loss for the first class as representative
        os.makedirs("results", exist_ok=True)
        first_cls = next(iter(history))
        losses, val_losses = history[first_cls]
        plt.figure()
        plt.plot(losses, label="train_loss")
        if val_losses is not None:
            plt.plot(val_losses, label="val_loss")
        plt.title(f"Loss curve ({name})")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join("results", f"loss_{name}.png"))
        plt.close()

        print(f"Results for {name}: scratch acc={metrics['accuracy']:.4f}, sklearn acc={sk_metrics['accuracy']:.4f}")

    # Save a summary CSV
    import json

    # Convert numpy arrays in history to lists for JSON serialization
    serializable = {}
    for name, info in results.items():
        info_copy = {k: v for k, v in info.items() if k != "history"}
        hist = info.get("history", {})
        hist_serial = {}
        for cls, (losses, val_losses) in hist.items():
            hist_serial[str(cls)] = {
                "losses": losses.tolist() if hasattr(losses, "tolist") else losses,
                "val_losses": (val_losses.tolist() if (hasattr(val_losses, "tolist") and val_losses is not None) else None),
            }
        info_copy["history"] = hist_serial
        serializable[name] = info_copy

    with open(os.path.join("results", "summary.json"), "w") as f:
        json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    run()
