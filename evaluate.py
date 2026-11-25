"""Evaluate the models on their held-out test sets, float vs INT8.
Writes charts (PNG) and summary.csv to ./results.

    python evaluate.py            # every model that has a cache (eye, mouth)
    python evaluate.py eye        # only the eye model (keeps the mouth rows)
    python evaluate.py mouth      # only the mouth model (keeps the eye rows)
"""
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)

import common
import config

import keras

VARIANTS = {            # label -> model file variant
    "float (Keras)": "keras",
    "float (TFLite)": "float",
    "INT8 (TFLite)": "int8",
}


def predict_all(variant, name, X):
    """Returns (P(positive) per image, avg single-image inference time in ms)."""
    path = common.model_path(name, variant)
    if variant == "keras":
        model = keras.models.load_model(path)
        run = lambda img: model(img.reshape(1, *img.shape, 1).astype(np.float32),
                                training=False).numpy()[0]
    else:
        run = common.TFLiteModel(path).predict
    for img in X[:5]:                                  # warm-up
        run(img)
    probs, start = [], time.perf_counter()
    for img in X:
        probs.append(run(img)[1])
    ms = (time.perf_counter() - start) / len(X) * 1000
    return np.array(probs), ms


def plot_training_curves(name):
    hist = json.loads((config.RESULTS_DIR / f"{name}_history.json").read_text())
    epochs = range(1, len(hist["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(epochs, hist[key], label="training")
        ax.plot(epochs, hist[f"val_{key}"], label="validation")
        ax.set(title=f"{name} model - {title}", xlabel="epoch", ylabel=title.lower())
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"{name}_training_curves.png", dpi=150)
    plt.close(fig)


def evaluate_model(name):
    classes = config.MODELS[name]                      # (negative, positive)
    X, y = common.load_split(name)["test"]
    print(f"[{name}] test set: {len(y)} images "
          f"({classes[0]} {int((y == 0).sum())}, {classes[1]} {int((y == 1).sum())})")
    rows, cms, rocs = [], {}, {}
    for label, variant in VARIANTS.items():
        probs, ms = predict_all(variant, name, X)
        pred = (probs >= 0.5).astype(int)              # = argmax of the 2 outputs
        fpr, tpr, _ = roc_curve(y, probs)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        cms[label] = cm
        rocs[label] = (fpr, tpr, auc(fpr, tpr))
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        rows.append({
            "model": name, "variant": label, "positive_class": classes[1],
            "test_images": len(y),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": rocs[label][2],
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "size_kb": common.model_path(name, variant).stat().st_size / 1024,
            "avg_inference_ms": ms,
        })

    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 4))
    for ax, (label, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=classes, yticklabels=classes,
               xlabel="predicted", ylabel="actual", title=f"{name} - {label}")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for (label, (fpr, tpr, area)), style in zip(rocs.items(), ("-", "--", ":")):
        ax.plot(fpr, tpr, style, linewidth=2, label=f"{label} (AUC = {area:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=0.8)
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title=f"{name} model - ROC ({classes[1]} = positive)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"{name}_roc.png", dpi=150)
    plt.close(fig)

    if (config.RESULTS_DIR / f"{name}_history.json").exists():
        plot_training_curves(name)
    return rows


def compare_yawn_methods():
    """Per-photo yawn decision on the mouth test set: MAR alone, the mouth CNN
    alone, and both agreeing (the rule the demo uses when YAWN_USE_CNN is on).
    MAR uses the default (uncalibrated) threshold."""
    split = common.load_split("mouth")
    X, y = split["test"]
    mar = split["mar"]["test"]
    threshold = max(config.MAR_BASELINE_DEFAULT + config.MAR_YAWN_OFFSET,
                    config.MAR_THRESHOLD_MIN)
    model = common.TFLiteModel(common.model_path("mouth", "int8"))
    cnn = np.array([model.predict(img)[1] for img in X]) >= config.YAWN_CNN_MIN
    by_mar = mar >= threshold
    rows = []
    for label, pred in ((f"MAR >= {threshold:.2f}", by_mar), ("mouth CNN (INT8)", cnn),
                        ("MAR and CNN agree", by_mar & cnn)):
        pred = pred.astype(int)
        rows.append({
            "method": label, "test_images": len(y),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "false_yawns": int(((pred == 1) & (y == 0)).sum()),
            "missed_yawns": int(((pred == 0) & (y == 1)).sum()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "yawn_methods.csv", index=False, float_format="%.4f")
    return df


def has_cache(name):
    return (config.CACHE_DIR / f"{name}.npz").exists()


if __name__ == "__main__":
    config.RESULTS_DIR.mkdir(exist_ok=True)
    requested = sys.argv[1:]
    unknown = [n for n in requested if n not in config.MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}. Choose from {list(config.MODELS)}.")
    names = requested or [n for n in config.MODELS if has_cache(n)]
    if not names:
        raise SystemExit("No cached data found. Run train.py first.")
    for n in names:
        if not has_cache(n):
            raise SystemExit(f"cache\\{n}.npz is missing. Run: python train.py {n}")

    all_rows = []
    for model_name in names:
        all_rows += evaluate_model(model_name)
    new = pd.DataFrame(all_rows)

    # keep the rows of models that were not evaluated this time (e.g. the mouth
    # rows from a teammate's earlier run) instead of overwriting the file
    summary_path = config.RESULTS_DIR / "summary.csv"
    df = new
    if summary_path.exists():
        try:
            old = pd.read_csv(summary_path)
            old = old[~old["model"].isin(names)]
            df = pd.concat([old, new], ignore_index=True)
        except Exception as err:
            print(f"Could not merge with the existing summary.csv ({err}); overwriting it.")
    order = {n: i for i, n in enumerate(config.MODELS)}
    df = df.sort_values("model", key=lambda s: s.map(order), kind="stable")
    df.to_csv(summary_path, index=False, float_format="%.4f")

    pd.set_option("display.width", 220)
    print(new.round(4).to_string(index=False))
    if "mouth" in names:
        print("\nYawn decision per photo (mouth test set):")
        print(compare_yawn_methods().round(4).to_string(index=False))
    print(f"\nSaved charts and summary.csv in {config.RESULTS_DIR}")