from __future__ import annotations
import os
import time
import contextlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from . import config, utils, data, model as mdl

from fruits360 import utils
utils.ensure_device(prefer_gpu=True)

from fruits360 import config
print(f"Input size: IMAGE_SIZE={config.IMAGE_SIZE}, INPUT_SHAPE={config.INPUT_SHAPE}")


def _plot_training_curves(history: keras.callbacks.History) -> None:
    acc_path = Path(getattr(config, "PLOT_ACC_PATH", config.ARTIFACTS / "acc.png"))
    loss_path = Path(getattr(config, "PLOT_LOSS_PATH", config.ARTIFACTS / "loss.png"))
    acc_path.parent.mkdir(parents=True, exist_ok=True)

    hist = history.history
    epochs = range(1, len(hist.get("accuracy", [])) + 1)

    # Accuracy
    plt.figure(figsize=(7, 5))
    for key in [k for k in hist.keys() if k.lower() in {"accuracy", "val_accuracy"}]:
        plt.plot(epochs, hist[key], label=key)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy")
    plt.grid(True, ls="--", alpha=.4); plt.legend(); plt.tight_layout()
    plt.savefig(acc_path, dpi=150); plt.close()

    # Loss
    plt.figure(figsize=(7, 5))
    for key in [k for k in hist.keys() if k.lower() in {"loss", "val_loss"}]:
        plt.plot(epochs, hist[key], label=key)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.grid(True, ls="--", alpha=.4); plt.legend(); plt.tight_layout()
    plt.savefig(loss_path, dpi=150); plt.close()

    print(f"[plot] Wrote:\n - {acc_path}\n - {loss_path}")


def _maybe_make_alias(primary_csv: Path) -> None:
    """
    If a second log filename is configured and would contain the exact same data,
    create/refresh a symlink that points to the primary CSV instead of writing twice.
    """
    alias_path = getattr(config, "TRAIN_LOG_CSV", None)
    if not alias_path:
        # Back-compat: some configs may use HISTORY_LOG_CSV or HISTORY_CSV
        alias_path = getattr(config, "HISTORY_LOG_CSV", None)
    if not alias_path:
        return

    alias_path = Path(alias_path)
    if alias_path.resolve() == primary_csv.resolve():
        return  # same file already; nothing to do

    alias_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if alias_path.exists() or alias_path.is_symlink():
            alias_path.unlink()
        alias_path.symlink_to(primary_csv.resolve())
        print(f"[log] Alias created: {alias_path} -> {primary_csv.name}")
    except Exception as e:
        # Symlink may be restricted; silently fall back by copying once at end.
        try:
            import shutil
            shutil.copy2(primary_csv, alias_path)
            print(f"[log] Copied log once to: {alias_path} (symlink unavailable: {e})")
        except Exception as e2:
            print(f"[log] Could not create alias for log file: {e2}")


def main():
    print("TF:", tf.__version__)
    utils.tune_threads()

    if os.environ.get("FRUITS360_FORCE_CPU", "0") == "1":
        utils.force_cpu_only()
        print("Running on CPU (forced via FRUITS360_FORCE_CPU=1)")
    else:
        print("GPUs:", tf.config.list_physical_devices("GPU"))

    # Data
    train_ds, val_ds, class_names = data.load_train_val()
    import json
    from . import config
#   save class names for inference display.
#   with open(config.ARTIFACTS / "class_names.json", "w") as f:
#        json.dump(class_names, f, indent=2)
    cls_json = config.ARTIFACTS / "class_names.json"
    cls_json.parent.mkdir(parents=True, exist_ok=True)
    with open(cls_json, "w") as f:
        import json
        json.dump(class_names, f, indent=2)
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    # Model
    net = mdl.build_model(num_classes)

    # SINGLE CSV logger path (avoid duplicates)
    primary_csv = Path(getattr(config, "HISTORY_CSV", config.ARTIFACTS / "train_log.csv"))
    primary_csv.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=config.BEST_KERAS,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=3,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(primary_csv.as_posix()),
    ]

    # Train
    t0 = time.time()
    history = net.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        verbose=1,
        callbacks=callbacks,
    )
    t1 = time.time()
    print(
        f"Training done in {(t1 - t0)/60:.1f} min; "
        f"best val_acc={max(history.history['val_accuracy']):.4f}"
    )

    # Save best model (weights already restored to best by EarlyStopping)
    net.save(config.BEST_KERAS)
    print(f"Saved Keras model -> {config.BEST_KERAS}")

    # Optional SavedModel export (ONCE)
    if getattr(config, "EXPORT_SAVEDMODEL", True):
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            net.export(str(config.SAVEDMODEL_DIR))
        devnull.close()
        print(f"Exported SavedModel -> {config.SAVEDMODEL_DIR}")

    # Plots
    _plot_training_curves(history)

    # If another history filename is configured, make it a symlink/copy to avoid duplicate writing
    _maybe_make_alias(primary_csv)


if __name__ == "__main__":
    main()
