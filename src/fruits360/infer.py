# src/fruits360/infer.py
from __future__ import annotations

import os, sys, json, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quiet TF

from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Project modules
from fruits360 import config, utils

# One-time runtime setup
utils.tune_threads()
utils.ensure_device(prefer_gpu=True)

# ---- Preprocessing and model registry ----
# Our custom MobileNetV2 preprocess (registered via @keras.saving.register_keras_serializable)
from fruits360.model import mobilenetv2_preprocess as _registered_mb_pre
# Keras' raw MobileNetV2 preprocess (used only if model lacks a preprocess layer)
from keras.applications.mobilenet_v2 import preprocess_input as _raw_mb_pre

# ---- Class names (prefer the saved order from training) ----
cls_path = Path(config.ARTIFACTS) / "class_names.json"
if cls_path.exists():
    try:
        class_names = json.loads(cls_path.read_text())
        print(f"[infer] Loaded class names from {cls_path} ({len(class_names)} classes)")
    except Exception as e:
        print(f"[infer] Failed to load class_names.json: {e}; scanning TRAIN_DIR instead.")
        class_names = sorted([d.name for d in Path(config.TRAIN_DIR).iterdir() if d.is_dir()])
else:
    class_names = sorted([d.name for d in Path(config.TRAIN_DIR).iterdir() if d.is_dir()])
    print(f"[infer] Using class names from TRAIN_DIR ({len(class_names)} classes)")

# ---------------- argparse ----------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fruits-360 inference")
    p.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to an image file. If omitted, the first *.jpg under TEST_DIR is used.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=5,
        help="How many top predictions to display.",
    )
    return p.parse_args()

# ---------------- model loading ----------------
def _load_model() -> keras.Model | keras.layers.Layer:
    best = Path(config.BEST_KERAS)
    saved = Path(config.SAVEDMODEL_DIR)

    if best.exists():
        try:
            print(f"[infer] Loading .keras model: {best}")
            return keras.models.load_model(
                best,
                compile=False,
                custom_objects={
                    "fruits360>mobilenetv2_preprocess": _registered_mb_pre,
                    "mobilenetv2_preprocess": _registered_mb_pre,
                    "preprocess_input": _raw_mb_pre,
                },
            )
        except Exception as e:
            print(f"[infer] .keras load failed, will try SavedModel:\n{e}", file=sys.stderr)

    if saved.exists():
        print(f"[infer] Loading SavedModel via TFSMLayer: {saved}")
        return keras.layers.TFSMLayer(str(saved), call_endpoint="serving_default")

    raise FileNotFoundError(f"No trained model found:\n - {best}\n - {saved}")

# ---------------- class names fallback helpers ----------------
def _scan_class_names(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def _get_class_names() -> List[str]:
    try:
        if class_names:
            return class_names
    except NameError:
        pass
    for root in (Path(getattr(config, "TRAIN_DIR", "")), Path(getattr(config, "TEST_DIR", ""))):
        names = _scan_class_names(root)
        if names:
            return names
    return []

# ---------------- model preprocess detection ----------------
def _model_has_preprocess(model: keras.Model | keras.layers.Layer) -> bool:
    if hasattr(model, "layers"):
        for l in model.layers:
            name = getattr(l, "name", "").lower()
            if "preprocess" in name or "mobilenetv2" in name:
                return True
    return False  # TFSMLayer unknown; treat as False

# ---------------- image IO: letterbox pad + resize ----------------
def _letterbox_pad_to_square_pil(img: "PIL.Image.Image") -> "PIL.Image.Image":
    from PIL import ImageOps
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    dw, dh = side - w, side - h
    padding = (dw // 2, dh // 2, dw - (dw // 2), dh - (dh // 2))
    return ImageOps.expand(img, border=padding, fill=0)

def _load_and_prepare_image(img_path: Path, apply_preprocess: bool) -> np.ndarray:
    pil = keras.utils.load_img(img_path)  # RGB PIL
    pil = _letterbox_pad_to_square_pil(pil)
    hw = config.IMAGE_SIZE[:2]
    pil = pil.resize(hw)
    x = keras.utils.img_to_array(pil)  # float32 0..255
    x = np.expand_dims(x, 0)           # (1,H,W,3)
    if apply_preprocess:
        x = _raw_mb_pre(x.copy())      # MobileNetV2 preprocess
    return x

# ---------------- prediction ----------------
def _predict(model: keras.Model | keras.layers.Layer, x: np.ndarray) -> np.ndarray:
    out = model(x, training=False)
    if isinstance(out, dict):
        out = next(iter(out.values()))
    p = tf.convert_to_tensor(out).numpy().squeeze()

    # If not already probabilities, softmax them
    prob_sum = float(np.sum(p))
    if not (0.999 <= prob_sum <= 1.001):
        p = tf.nn.softmax(p).numpy()
    return p

# ---------------- main ----------------
def main():
    args = _parse_args()
    print("GPUs visible:", tf.config.list_physical_devices("GPU"))

    model = _load_model()
    names = _get_class_names()

    # Resolve image path
    img_path: Optional[Path] = None
    if args.image:
        img_path = Path(args.image).expanduser().resolve()
        if not img_path.exists():
            print(f"[infer] Image not found: {img_path}", file=sys.stderr)
            sys.exit(2)
    else:
        test_root = Path(getattr(config, "TEST_DIR", ""))
        img_path = next(test_root.rglob("*.jpg"), None) if test_root.exists() else None
        if img_path is None:
            print(f"[infer] No image provided and no .jpg found under TEST_DIR={test_root}", file=sys.stderr)
            sys.exit(2)

    needs_pp = not _model_has_preprocess(model)
    print(f"[infer] Model has built-in preprocess: {not needs_pp}")

    x = _load_and_prepare_image(img_path, apply_preprocess=needs_pp)
    probs = _predict(model, x)

    print(f"\nImage: {img_path}")
    print(f"[debug] prob_sum={probs.sum():.6f} top1={probs.max():.4f}")

    k = max(1, min(args.topk, probs.shape[-1]))
    top = np.argsort(-probs)[:k]
    print(f"Top-{k} predictions:")
    for r, idx in enumerate(top, 1):
        label = names[idx] if idx < len(names) and names else f"class_{idx}"
        print(f"  {r:>2}. {label:<32}  {probs[idx]:.4f}")

if __name__ == "__main__":
    main()
