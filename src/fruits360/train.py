#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train script for Fruits360 project — CPU/GPU auto-configured.
Works on macOS (Metal), Ubuntu/Linux (CUDA/CPU), and Windows.
"""

import os
import json
import inspect
import multiprocessing
import platform

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress INFO/WARNING C++ logs of the following Apple Metal behavior
'''
I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
'''
# ============================================================
# 0) Automatic CPU Thread Scaling (~75% utilization)
#    (set BEFORE importing TensorFlow so TF picks them up)
# ============================================================
# base 0.75 or overwrite
_cpu = multiprocessing.cpu_count()
_scale = float(os.environ.get("FRUITS360_THREAD_SCALE", "0.75"))
_omp   = max(1, int(_cpu * _scale))
_intra = _omp
_inter = max(1, _cpu // 4)
#excplict manual overwrite(if exported)
_omp   = int(os.environ.get("FRUITS360_OMP_THREADS",   _omp))
_intra = int(os.environ.get("FRUITS360_TF_INTRAOP_THREADS", _intra))
_inter = int(os.environ.get("FRUITS360_TF_INTEROP_THREADS", _inter))

os.environ["OMP_NUM_THREADS"] = str(_omp)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(_intra)
os.environ["TF_NUM_INTEROP_THREADS"] = str(_inter)

print(f"[Auto Thread Config] Detected {_cpu} cores → OMP={_omp}, INTRA={_intra}, INTER={_inter}")

# Now import TF & project modules
import tensorflow as tf  # noqa: E402
from fruits360 import data, model, config  # noqa: E402
# --- Optional: enable mixed precision only on GPU ---
from tensorflow.keras import mixed_precision

if tf.config.list_physical_devices("GPU"):
    mixed_precision.set_global_policy("mixed_float16")
    print("[policy] Mixed precision enabled (float16 on GPU)")
else:
    mixed_precision.set_global_policy("float32")
    print("[policy] CPU only → using float32 (safe default)")


# Optional seeding helper (use if available)
try:
    from fruits360.utils import setup_seed  # noqa: E402
except Exception:
    import random
    import numpy as np

    def setup_seed(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print(f"[utils] Random seed set to {seed}")

# ============================================================
# 1) GPU memory configuration (prevent desktop freeze)
# ============================================================
_gpus = tf.config.list_physical_devices("GPU")
if _gpus:
    try:
        for g in _gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[GPU] Memory growth enabled for {len(_gpus)} GPU(s)")
    except Exception as e:
        print(f"[GPU] Warning: could not enable memory growth: {e}")
else:
    print("[GPU] No GPUs visible; CPU mode active")

# ============================================================
# 2) Helpers
# ============================================================
def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _auto_runtime_setup() -> int:
    """
    Configure runtime parameters automatically depending on GPU/CPU environment.
    Returns the chosen batch size.
    """
    env_bs = os.environ.get("FRUITS360_BATCH_SIZE")
    if env_bs:
        bs = int(env_bs)
        print(f"[train] Batch size overridden by FRUITS360_BATCH_SIZE={bs}")
        return bs
    num_gpus = len(tf.config.list_physical_devices("GPU"))
    print(f"[Runtime] GPUs visible: {tf.config.list_physical_devices('GPU')}")
    print(f"Input size: IMAGE_SIZE={config.IMAGE_SIZE}, INPUT_SHAPE={config.INPUT_SHAPE}")
    print(f"TF: {tf.__version__}")

    # Inline heuristic for batch size (no config.suggest_batch_size dependency)
    if num_gpus >= 1:
        if _is_apple_silicon():
            bs = 64     # good default for Apple Metal GPU
        else:
            bs = 128    # typical CUDA/ROCm GPU
    else:
        logical = os.cpu_count() or 8
        bs = max(8, min(32, (int(logical * 0.75) // 2) * 2))  # even number 8–32

    print(f"[train] Auto batch size set to: {bs}")
    return bs


def _build_model_adaptive(num_classes: int):
    """
    Build the model by mapping common argument names to whatever
    your model.build_model actually accepts.
    """
    fn = model.build_model
    params = set(inspect.signature(fn).parameters.keys())

    candidates = {
        "num_classes": num_classes,
        "n_classes": num_classes,
        "classes": num_classes,

        "input_shape": config.INPUT_SHAPE,   # (H,W,C) e.g. (224,224,3)
        "img_size":   config.IMAGE_SIZE,     # (H,W)
        "input_size": config.IMAGE_SIZE,

        "backbone": config.BACKBONE,
        "base":     config.BACKBONE,

        "pretrained": config.PRETRAINED,
        "weights":   ("imagenet" if config.PRETRAINED else None),

        "dropout":      config.DROPOUT,
        "dropout_rate": config.DROPOUT,

        "pooling":     config.GLOBAL_POOL,
        "global_pool": config.GLOBAL_POOL,

        "freeze":        config.FREEZE,
        "freeze_base":   config.FREEZE,
        "frozen_layers": config.FROZEN_LAYERS,

        "classifier_activation": config.CLASSIFIER_ACT,
        "activation":            config.CLASSIFIER_ACT,
    }

    kwargs = {k: v for k, v in candidates.items() if (k in params and v is not None)}
    try:
        return fn(**kwargs)
    except TypeError:
        # If the function is very custom or positional-only, fall back.
        return fn()


# ============================================================
# 3) Main training routine
# ============================================================
def main():
    # Reproducibility
    setup_seed(config.SEED)

    # Auto runtime setup (threads already set; decide batch size)
    batch_size = _auto_runtime_setup()

    # Make batch size visible to anything reading config/env
    config.BATCH_SIZE = batch_size
    config.BATCH = batch_size
    config.EVAL_BATCH_SIZE = max(batch_size, getattr(config, "EVAL_BATCH_SIZE", batch_size))
    os.environ["FRUITS360_BATCH_SIZE"] = str(batch_size)

    # Load datasets (data.py reads config.BATCH_SIZE internally)
    train_ds, val_ds, class_names = data.load_train_val()

    # Save class names for inference
    with open(config.CLASS_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Build & compile model
    net = _build_model_adaptive(num_classes=len(class_names))
    # Label smoothing for better generalization to improve accuracy gap between train and eval
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.1)
    )
    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"],
        steps_per_execution=64,   # safe on CPU & GPU; fewer Python callbacks
    )

    # Callbacks
    cbs = config.default_callbacks()

    # Train
    history = net.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=cbs,
        verbose=config.VERBOSE,
    )

    # Save final/best model
    net.save(config.BEST_KERAS)
    print(f"[train] Saved best model → {config.BEST_KERAS}")

    # --- ensure the .keras file is fully written and visible ---
    import time
    best_path = str(config.BEST_KERAS)
    if os.path.exists(best_path):
        try:
            # force flush-to-disk on filesystems that delay writes
            with open(best_path, "ab") as f:
                f.flush()
                os.fsync(f.fileno())
            print(f"[train] Verified & fsync'd best model: {best_path}")
        except Exception as e:
            print(f"[train] Warning: fsync failed ({e})")
    else:
        # if the save is slightly asynchronous, give it a short window
        print("[train] Best model not visible yet; waiting 2s...")
        time.sleep(2)
        if os.path.exists(best_path):
            print(f"[train] Found best model after delay: {best_path}")
        else:
            print(f"[train] Error: best model still missing after delay.")

    # Report best metric
    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    best_epoch = history.history.get("val_accuracy", [0.0]).index(best_val_acc) + 1
    print(f"[train] Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")

    # Plots
    try:
        config.plot_history()
        print("[plot] Training curves saved")
    except Exception:
        pass

# ============================================================
# 4) Entry point
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[train] Interrupted by user.")
    except Exception:
        import traceback
        print("[train] Exception occurred:\n", traceback.format_exc())
