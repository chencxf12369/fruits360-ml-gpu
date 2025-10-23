#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train script for Fruits360 project — CPU/GPU auto-configured.
Compatible with macOS (Metal), Ubuntu (CUDA), and Windows (CPU).
"""

import os
import multiprocessing
import platform
import tensorflow as tf
from fruits360 import data, model, config
from fruits360.utils import setup_seed


# ============================================================
# 1. Automatic CPU Thread Scaling (~75% utilization)
# ============================================================
cpu_cores = multiprocessing.cpu_count()
omp = max(1, int(cpu_cores * 0.75))
intra = omp
interop = max(1, cpu_cores // 4)

os.environ["OMP_NUM_THREADS"] = str(omp)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(intra)
os.environ["TF_NUM_INTEROP_THREADS"] = str(interop)

print(f"[Auto Thread Config] Detected {cpu_cores} cores → "
      f"OMP={omp}, INTRA={intra}, INTER={interop}")

# ============================================================
# 2. GPU memory configuration (prevent freeze)
# ============================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"[GPU] Memory growth enabled for {len(gpus)} GPU(s)")
    except Exception as e:
        print(f"[GPU] Warning: could not enable memory growth: {e}")
else:
    print("[GPU] No GPUs visible; CPU mode active")

# ============================================================
# 3. Helpers
# ============================================================
def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def _auto_runtime_setup():
    """
    Configure runtime parameters automatically depending on GPU/CPU environment.
    """
    num_gpus = len(tf.config.list_physical_devices("GPU"))
    print(f"[Runtime] GPUs visible: {tf.config.list_physical_devices('GPU')}")
    print(f"Input size: IMAGE_SIZE={config.IMAGE_SIZE}, INPUT_SHAPE={config.INPUT_SHAPE}")
    print(f"TF: {tf.__version__}")

    # ---- Inline replacement for config.suggest_batch_size() ----
    if num_gpus >= 1:
        # GPU detected → bigger batch
        if _is_apple_silicon():
            bs = 64     # good default for Apple Metal GPU
        else:
            bs = 128    # typical CUDA/ROCm GPU
    else:
        # CPU only → balanced at ~75% of logical cores
        logical = os.cpu_count() or 8
        bs = max(8, min(32, (int(logical * 0.75) // 2) * 2))  # even number 8–32
    # ------------------------------------------------------------

    print(f"[train] Auto batch size set to: {bs}")
    return bs


# ============================================================
# 4. Main training routine
# ============================================================
def main():
    # Seed control for reproducibility
    setup_seed(config.SEED)

    # Auto runtime setup (threads, batch size, etc.)
    batch_size = _auto_runtime_setup()

    # Load datasets
    train_ds, val_ds, class_names = data.load_train_val(batch_size=batch_size)

    # Save class names for inference
    import json
    with open(config.CLASS_NAMES_JSON, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # Build model
    net = model.build_model(
        input_shape=config.INPUT_SHAPE,
        num_classes=len(class_names),
        backbone=config.BACKBONE,
        pretrained=config.PRETRAINED,
        dropout=config.DROPOUT,
        pooling=config.GLOBAL_POOL,
        freeze=config.FREEZE,
        frozen_layers=config.FROZEN_LAYERS,
        classifier_activation=config.CLASSIFIER_ACT,
    )

    # Compile
    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Prepare callbacks
    cbs = config.default_callbacks()

    # Fit model
    history = net.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=cbs,
        verbose=config.VERBOSE,
    )

    # Save final model
    net.save(config.BEST_KERAS)
    print(f"[train] Saved best model → {config.BEST_KERAS}")

    # Plot results
    try:
        config.plot_history()
        print("[plot] Training curves saved")
    except Exception:
        pass


# ============================================================
# 5. Entry point
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[train] Interrupted by user.")
    except Exception as e:
        import traceback
        print("[train] Exception occurred:\n", traceback.format_exc())
