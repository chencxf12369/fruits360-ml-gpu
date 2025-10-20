# src/fruits360/eval.py
from __future__ import annotations
import json
from pathlib import Path
import os
from typing import Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless for macOS/CI
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

from . import config, utils, data
from fruits360 import utils
utils.ensure_device(prefer_gpu=True)

# Try to import registered preprocess for Keras deserialization
try:
    from .model import mobilenetv2_preprocess  # registered via @keras.saving.register_keras_serializable("fruits360")
except Exception:
    mobilenetv2_preprocess = None


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _class_names_from_dir_or_cache(root: Path) -> list[str]:
    """Prefer cached class_names.json; fallback to scanning directory."""
    cache_file = config.ARTIFACTS / "class_names.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                names = json.load(f)
            if isinstance(names, list) and names:
                return names
        except Exception:
            pass
    # fallback: infer from directories
    names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    try:
        config.save_class_names(names)
    except Exception:
        pass
    return names


def _plot_eval_summary(acc: float, loss: float) -> None:
    """Simple bar plot of accuracy and loss."""
    out_path = Path(getattr(config, "PLOT_EVAL_PATH", config.ARTIFACTS / "eval_summary.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    labels = ["accuracy", "loss"]
    values = [acc, loss]
    plt.bar(labels, values, color=["steelblue", "salmon"])
    plt.ylim(0, max(1.0, max(values) * 1.1))
    plt.title("Evaluation Summary")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Wrote: {out_path}")


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """Plot normalized confusion matrix with colorbar."""
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    out_path = Path(getattr(config, "PLOT_CM_PATH", config.ARTIFACTS / "confusion_matrix.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix (normalized)")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Wrote: {out_path}")


# ---------------------------------------------------------------------
# Model loading (same structure, no loss)
# ---------------------------------------------------------------------
def _load_model_or_tfsmlayer() -> Tuple[Union[keras.Model, None], Union[keras.layers.Layer, None]]:
    """Try to load .keras with custom_objects; fallback to TFSMLayer."""
    ko = {}
    if mobilenetv2_preprocess is not None:
        ko["fruits360>mobilenetv2_preprocess"] = mobilenetv2_preprocess
        ko["mobilenetv2_preprocess"] = mobilenetv2_preprocess
    try:
        mdl = keras.models.load_model(Path(config.BEST_KERAS), custom_objects=ko or None)
        print(f"Loaded model: {config.BEST_KERAS}")
        return mdl, None
    except Exception as e:
        print(f"Failed to load .keras model, trying SavedModel: {e}")

    try:
        layer = keras.layers.TFSMLayer(str(config.SAVEDMODEL_DIR), call_endpoint="serving_default")
        print(f"Loaded SavedModel via TFSMLayer: {config.SAVEDMODEL_DIR}")
        return None, layer
    except Exception as e:
        raise RuntimeError(f"Could not load model from either .keras or SavedModel: {e}") from e


# ---------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------
def _evaluate_keras_model(model: keras.Model, test_ds) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate a native Keras model."""
    results = model.evaluate(test_ds, verbose=1)
    if isinstance(results, (list, tuple)) and len(results) >= 2:
        metrics = {"loss": float(results[0]), "accuracy": float(results[1])}
    elif isinstance(results, dict):
        metrics = {k: float(v) for k, v in results.items()}
    else:
        metrics = {"loss": float(results)}

    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=-1)

    y_true = []
    for _, labels in test_ds:
        y_true.append(tf.argmax(labels, axis=-1).numpy())
    y_true = np.concatenate(y_true, axis=0)

    metrics["n_samples"] = len(y_true)
    return metrics, y_true, y_pred


def _evaluate_tfsmlayer(layer: keras.layers.Layer, test_ds) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate SavedModel through TFSMLayer inference."""
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="sum")
    total_loss = 0.0
    n_samples = 0
    n_correct = 0

    y_true_all, y_pred_all = [], []

    for images, labels in test_ds:
        outputs = layer(images, training=False)
        if isinstance(outputs, dict):
            outputs = next(iter(outputs.values()))
        outputs = tf.convert_to_tensor(outputs)
        batch_size = int(images.shape[0])

        total_loss += float(cce(labels, outputs).numpy())
        pred_idx = tf.argmax(outputs, axis=-1).numpy()
        true_idx = tf.argmax(labels, axis=-1).numpy()
        n_correct += int((pred_idx == true_idx).sum())
        n_samples += batch_size
        y_true_all.append(true_idx)
        y_pred_all.append(pred_idx)

    loss = total_loss / max(n_samples, 1)
    acc = n_correct / max(n_samples, 1)
    metrics = {"loss": float(loss), "accuracy": float(acc), "n_samples": int(n_samples)}

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    return metrics, y_true, y_pred


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    utils.tune_threads()
    print("GPUs visible:", tf.config.list_physical_devices("GPU"))

    keras_model, tfsmlayer = _load_model_or_tfsmlayer()

    # Use same test loader as train pipeline (handles pad/resize)
    test_ds = data.load_test()

    if keras_model is not None:
        metrics, y_true, y_pred = _evaluate_keras_model(keras_model, test_ds)
    else:
        metrics, y_true, y_pred = _evaluate_tfsmlayer(tfsmlayer, test_ds)

    # Write metrics JSON
    metrics_path = config.ARTIFACTS / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Eval metrics:", metrics)
    print(f"[saved] {metrics_path}")

    # Plot summary
    _plot_eval_summary(metrics.get("accuracy", 0.0), metrics.get("loss", 0.0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = _class_names_from_dir_or_cache(Path(config.TEST_DIR))
    _plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()