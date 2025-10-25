# src/fruits360/eval.py
from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF

import json, gc, sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

from . import config, data, utils

# Optional: force float32 for eval stability on Metal
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")
except Exception:
    pass

# Custom preprocess for deserialization (if registered)
try:
    from .model import mobilenetv2_preprocess
except Exception:
    mobilenetv2_preprocess = None

# ---------------- plots ----------------
def _plot_eval_summary(acc: float, loss: float) -> None:
    out_path = Path(getattr(config, "PLOT_EVAL_PATH", config.ARTIFACTS / "evaluation_metrics.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["accuracy", "loss"], [acc, loss])
    plt.ylim(0, max(1.0, max(acc, loss) * 1.1))
    plt.title("Evaluation Summary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Wrote: {out_path}", flush=True)

def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    out_path = Path(getattr(config, "PLOT_CM_PATH", config.ARTIFACTS / "confusion_matrix.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix (normalized)")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Wrote: {out_path}", flush=True)

# ---------------- model loading ----------------
def _load_model_or_tfsmlayer() -> Tuple[Union[keras.Model, None], Union[keras.layers.Layer, None]]:
    ko = {}
    if mobilenetv2_preprocess is not None:
        ko["fruits360>mobilenetv2_preprocess"] = mobilenetv2_preprocess
        ko["mobilenetv2_preprocess"] = mobilenetv2_preprocess
    try:
        mdl = keras.models.load_model(Path(config.BEST_KERAS), custom_objects=ko or None)
        print(f"Loaded model: {config.BEST_KERAS}", flush=True)
        return mdl, None
    except Exception as e:
        print(f"Failed to load .keras model, trying SavedModel: {e}", flush=True)
    try:
        layer = keras.layers.TFSMLayer(str(config.SAVEDMODEL_DIR), call_endpoint="serving_default")
        print(f"Loaded SavedModel via TFSMLayer: {config.SAVEDMODEL_DIR}", flush=True)
        return None, layer
    except Exception as e:
        raise RuntimeError(f"Could not load model from either .keras or SavedModel: {e}") from e

# ---------------- runners ----------------
def _evaluate_keras_model(model: keras.Model, test_ds):
    # Fast path: just evaluate (shows 355/355 bar) — no predict() here
    results = model.evaluate(test_ds, verbose=1)
    if isinstance(results, (list, tuple)) and len(results) >= 2:
        metrics = {"loss": float(results[0]), "accuracy": float(results[1])}
    elif isinstance(results, dict):
        metrics = {k: float(v) for k, v in results.items()}
    else:
        metrics = {"loss": float(results)}
    # n_samples (optional; cheap to compute from dataset cardinality if known)
    try:
        n = tf.data.experimental.cardinality(test_ds).numpy()
        if n < 0: raise ValueError
        metrics["n_samples"] = int(n)
    except Exception:
        pass
    return metrics

def _full_confusion_matrix(model_or_layer, test_ds, is_layer: bool):
    # Only if FRUITS360_EVAL_CM=1 — compute y_true/y_pred
    y_true, y_pred = [], []
    for images, labels in test_ds:
        if is_layer:
            outs = model_or_layer(images, training=False)
            if isinstance(outs, dict):
                outs = next(iter(outs.values()))
            probs = tf.convert_to_tensor(outs)
        else:
            probs = model_or_layer(images, training=False)
        y_pred.append(tf.argmax(probs, axis=-1).numpy())
        y_true.append(tf.argmax(labels, axis=-1).numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    cm = confusion_matrix(y_true, y_pred)
    return cm

# ---------------- entry ----------------
def main():
    utils.tune_threads()
    print("GPUs visible:", tf.config.list_physical_devices("GPU"), flush=True)

    keras_model, tfsmlayer = _load_model_or_tfsmlayer()
    test_ds = data.load_test()  # your pipeline

    # Fast eval: accuracy/loss only
    if keras_model is not None:
        metrics = _evaluate_keras_model(keras_model, test_ds)
        is_layer = False
        runner = keras_model
    else:
        # Light-weight manual loop for TFSMLayer (no second pass)
        m, t = _full_confusion_matrix, _evaluate_keras_model  # not used in fast path
        # For TFSMLayer we’ll do a minimal pass to get metrics
        # but you’re using .keras, so this branch likely won’t run
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="sum")
        total_loss = 0.0; n_samples = 0; n_correct = 0
        for images, labels in test_ds:
            outs = tfsmlayer(images, training=False)
            if isinstance(outs, dict):
                outs = next(iter(outs.values()))
            probs = tf.convert_to_tensor(outs)
            total_loss += float(cce(labels, probs).numpy())
            pi = tf.argmax(probs, axis=-1).numpy()
            ti = tf.argmax(labels, axis=-1).numpy()
            n_correct += int((pi == ti).sum())
            n_samples += int(images.shape[0])
        loss = total_loss / max(n_samples, 1)
        acc  = n_correct / max(n_samples, 1)
        metrics = {"loss": float(loss), "accuracy": float(acc), "n_samples": int(n_samples)}
        is_layer = True
        runner = tfsmlayer

    # Save metrics JSON
    metrics_path = config.ARTIFACTS / "eval_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Eval metrics:", metrics, flush=True)
    print(f"[saved] {metrics_path}", flush=True)

    # Summary plot (cheap)
    _plot_eval_summary(metrics.get("accuracy", 0.0), metrics.get("loss", 0.0))

    # Optional: full confusion matrix (opt-in; can take ~30s)
    #if os.environ.get("FRUITS360_EVAL_CM", "0") == "1":
    cm = _full_confusion_matrix(runner, test_ds, is_layer)
    class_names = sorted([d.name for d in Path(config.TEST_DIR).iterdir() if d.is_dir()])
    _plot_confusion_matrix(cm, class_names)

    # Clean exit
    try:
        del test_ds, keras_model, tfsmlayer
    except Exception:
        pass
    keras.backend.clear_session()
    sys.stdout.flush(); sys.stderr.flush()
    gc.collect()
    sys.exit(0)

if __name__ == "__main__":
    main()
