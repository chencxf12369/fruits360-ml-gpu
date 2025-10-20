# src/fruits360/config.py
from __future__ import annotations

import os
import json
import atexit
import pathlib
import platform
from datetime import datetime

# ----- Matplotlib (headless-safe) -----
import matplotlib
matplotlib.use("Agg")  # ensure file-only backend
import matplotlib.pyplot as plt

import pandas as pd

# Keras callbacks are optional; available for convenience.
try:
    from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
    _KERAS_OK = True
except Exception:
    _KERAS_OK = False


# ======================================================================================
# Project paths (no folder name changes)
# ======================================================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Artifacts directory under the top folder
ARTIFACTS = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ======================================================================================
# Dataset location policy (no rename)
# 1) FRUITS360_DATA env var
# 2) ~/Documents/ml-gpu/Fruit-Images-Dataset
# 3) ~/data/Fruit-Images-Dataset
# ======================================================================================
PROJECT_DATA = PROJECT_ROOT / "Fruit-Images-Dataset"
_env_path = os.environ.get("FRUITS360_DATA")

if _env_path:
    DATA_ROOT = pathlib.Path(_env_path)
elif PROJECT_DATA.exists():
    DATA_ROOT = PROJECT_DATA
else:
    DATA_ROOT = pathlib.Path.home() / "data" / "Fruit-Images-Dataset"

TRAIN_DIR = DATA_ROOT / "Training"
TEST_DIR  = DATA_ROOT / "Test"

# ======================================================================================
# Model I/O (stable names; no folder rename)
# ======================================================================================
BEST_KERAS      = ARTIFACTS / "fruits360_best.keras"
SAVEDMODEL_DIR  = ARTIFACTS / "fruits360_savedmodel"
EXPORT_SAVEDMODEL = os.environ.get("FRUITS360_EXPORT_SAVEDMODEL", "0") == "1"  # default off to avoid duplicates

HISTORY_CSV     = ARTIFACTS / "train_history.csv"         # single source of truth for training history
TRAIN_LOG       = HISTORY_CSV                             # backward-compatible alias (no second file)

CHECKPOINTS_DIR = ARTIFACTS / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATTERN = str(CHECKPOINTS_DIR / "ckpt_epoch{epoch:02d}_val{val_accuracy:.3f}.keras")

TENSORBOARD_LOGDIR = ARTIFACTS / "tb_logs"
TENSORBOARD_LOGDIR.mkdir(parents=True, exist_ok=True)

# Where plots will be written
PLOT_ACC_PATH   = ARTIFACTS / "training_accuracy.png"
PLOT_LOSS_PATH  = ARTIFACTS / "training_loss.png"
PLOT_BOTH_PATH  = ARTIFACTS / "training_curves.png"
PLOT_EVAL_PATH  = ARTIFACTS / "evaluation_metrics.png"

# Class names mapping (used by infer / eval pretty-print)
CLASS_NAMES_JSON = ARTIFACTS / "class_names.json"

# Evaluation metrics json (written by eval)
EVAL_JSON_PATH = ARTIFACTS / "eval_metrics.json"

RUN_TS = datetime.now().strftime("%Y%m%d-%H%M%S")

# ======================================================================================
# Training defaults
# ======================================================================================
SEED = int(os.environ.get("FRUITS360_SEED", "42"))

# Input size (can be overridden via env)
IMG_HEIGHT   = int(os.environ.get("FRUITS360_IMG_HEIGHT", "224"))
IMG_WIDTH    = int(os.environ.get("FRUITS360_IMG_WIDTH", "224"))
IMG_CHANNELS = int(os.environ.get("FRUITS360_IMG_CHANNELS", "3"))

# 2-tuple and 3-tuple shapes
IMAGE_SIZE  = (IMG_HEIGHT, IMG_WIDTH)                 # (H, W) — for dataset loader
IMG_SIZE    = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)   # (H, W, C) — for models
INPUT_SHAPE = IMG_SIZE

BATCH_SIZE = int(os.environ.get("FRUITS360_BATCH_SIZE", "64"))
EPOCHS     = int(os.environ.get("FRUITS360_EPOCHS", "30"))

USE_AUGMENTATION = os.environ.get("FRUITS360_USE_AUG", "1") == "1"
VALIDATION_SPLIT = float(os.environ.get("FRUITS360_VAL_SPLIT", "0.15"))

# Dataloader performance knobs
SHUFFLE_BUFFER = int(os.environ.get("FRUITS360_SHUFFLE_BUFFER", str(max(1000, BATCH_SIZE * 64))))
PREFETCH_AUTO  = os.environ.get("FRUITS360_PREFETCH_AUTO", "1") == "1"

# Square-pad settings for real-world photos/screenshots (applied in data.py if enabled)
PAD_TO_SQUARE  = os.environ.get("FRUITS360_PAD_TO_SQUARE", "1") == "1"   # keep aspect ratio, pad to square before resize
PAD_COLOR      = tuple(int(x) for x in os.environ.get("FRUITS360_PAD_COLOR", "0,0,0").split(","))  # RGB pad color
RESIZE_INTERP  = os.environ.get("FRUITS360_RESIZE_INTERP", "bilinear")   # bilinear|bicubic|nearest

# Eval / infer knobs
EVAL_BATCH_SIZE     = int(os.environ.get("FRUITS360_EVAL_BATCH", str(BATCH_SIZE)))
CONFUSION_NORMALIZE = os.environ.get("FRUITS360_CM_NORMALIZE", "true").lower() in {"1", "true", "yes"}
INFER_TOPK          = int(os.environ.get("FRUITS360_INFER_TOPK", "5"))
VERBOSE             = int(os.environ.get("FRUITS360_VERBOSE", "1"))

# ======================================================================================
# Threading defaults read by utils.tune_threads()
# ======================================================================================
_cpu = os.cpu_count() or 8
is_mac_arm = (platform.system() == "Darwin" and platform.machine() == "arm64")

if is_mac_arm:
    _default_omp   = min(8, _cpu)
    _default_intra = _default_omp
    _default_inter = max(1, _default_omp // 4)
else:
    _default_omp   = _cpu
    _default_intra = _cpu
    _default_inter = max(1, _cpu // 4)

OMP_THREADS         = int(os.environ.get("FRUITS360_OMP_THREADS", str(_default_omp)))
TF_INTRAOP_THREADS  = int(os.environ.get("FRUITS360_TF_INTRAOP_THREADS", str(_default_intra)))
TF_INTEROP_THREADS  = int(os.environ.get("FRUITS360_TF_INTEROP_THREADS", str(_default_inter)))

# Back-compat thread aliases
TF_INTRA = TF_INTRAOP_THREADS
TF_INTER = TF_INTEROP_THREADS
OMP      = OMP_THREADS

# ======================================================================================
# Model / transfer-learning settings (expected by model.py)
# ======================================================================================
BACKBONE      = os.environ.get("FRUITS360_BACKBONE", "mobilenetv2")  # mobilenetv2|resnet50|efficientnetb0
PRETRAINED    = os.environ.get("FRUITS360_PRETRAINED", "1") == "1"
GLOBAL_POOL   = os.environ.get("FRUITS360_GLOBAL_POOL", "avg")       # avg|max|none
DROPOUT       = float(os.environ.get("FRUITS360_DROPOUT", "0.2"))
LEARNING_RATE = float(os.environ.get("FRUITS360_LR", "0.001"))
WEIGHT_DECAY  = float(os.environ.get("FRUITS360_WEIGHT_DECAY", "0.0"))

FREEZE        = os.environ.get("FRUITS360_FREEZE", "1") == "1"
FROZEN_LAYERS = int(os.environ.get("FRUITS360_FROZEN_LAYERS", "0"))
CLASSIFIER_ACT = os.environ.get("FRUITS360_CLASSIFIER_ACT", "softmax")

# Back-compat model aliases
BASEMODEL      = BACKBONE
BASE_MODEL     = BACKBONE
BACKBONE_NAME  = BACKBONE
IMAGENET       = PRETRAINED
FREEZE_BASE    = FREEZE
FROZEN         = FREEZE
FREEZE_UP_TO   = FROZEN_LAYERS
DROPOUT_RATE   = DROPOUT
LR             = LEARNING_RATE
WD             = WEIGHT_DECAY
POOLING        = GLOBAL_POOL

# Back-compat training aliases
BATCH = BATCH_SIZE
EPOCH = EPOCHS
VAL_SPLIT = VALIDATION_SPLIT
USE_AUG   = USE_AUGMENTATION
HEIGHT    = IMG_HEIGHT
WIDTH     = IMG_WIDTH
CHANNELS  = IMG_CHANNELS
DATA_DIR  = DATA_ROOT
TRAIN_PATH = TRAIN_DIR
TEST_PATH  = TEST_DIR
ARTIFACT_DIR = ARTIFACTS
LOG_DIR      = TENSORBOARD_LOGDIR
CKPT_DIR     = CHECKPOINTS_DIR
BEST_MODEL   = BEST_KERAS
SAVEDMODEL   = SAVEDMODEL_DIR

# ======================================================================================
# Convenience: standard callbacks for training (optional to use)
# ======================================================================================
def default_callbacks(monitor: str = "val_accuracy"):
    """
    Returns a list of Keras callbacks that:
      - log epoch metrics to HISTORY_CSV (single file)
      - checkpoint the best model to BEST_KERAS (by monitor)
      - write TensorBoard logs under TENSORBOARD_LOGDIR
      - use ReduceLROnPlateau + EarlyStopping for stability
    Usage in train.py (optional):
        cb = config.default_callbacks()
        model.fit(..., callbacks=cb)
    """
    if not _KERAS_OK:
        return []

    callbacks = [
        CSVLogger(str(HISTORY_CSV), append=False),
        ModelCheckpoint(str(BEST_KERAS), save_best_only=True, monitor=monitor, mode="max"),
        TensorBoard(log_dir=str(TENSORBOARD_LOGDIR), update_freq="epoch"),
        ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor=monitor, patience=8, restore_best_weights=True, verbose=1),
    ]
    return callbacks

# ======================================================================================
# Class name helpers
# ======================================================================================
def save_class_names(class_names: list[str]) -> None:
    try:
        with open(CLASS_NAMES_JSON, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_class_names() -> list[str] | None:
    try:
        if CLASS_NAMES_JSON.exists():
            with open(CLASS_NAMES_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
    except Exception:
        pass
    return None

# ======================================================================================
# Plotting helpers (auto-run at exit if data is available)
# ======================================================================================
def plot_history(csv_path: pathlib.Path = HISTORY_CSV,
                 out_acc: pathlib.Path = PLOT_ACC_PATH,
                 out_loss: pathlib.Path = PLOT_LOSS_PATH,
                 out_both: pathlib.Path = PLOT_BOTH_PATH) -> bool:
    """
    Create accuracy and loss plots from a Keras CSVLogger history file.
    Returns True if plots were generated, else False.
    """
    try:
        if not csv_path.exists():
            return False
        df = pd.read_csv(csv_path)

        acc_cols  = [c for c in df.columns if c.lower() in {"accuracy", "acc"}]
        val_acc_cols = [c for c in df.columns if "val_" in c.lower() and ("accuracy" in c.lower() or c.lower() == "val_acc")]
        loss_cols = [c for c in df.columns if c.lower() == "loss"]
        val_loss_cols = [c for c in df.columns if c.lower() == "val_loss"]

        # Accuracy plot
        if acc_cols or val_acc_cols:
            plt.figure(figsize=(7, 5))
            for c in acc_cols:
                plt.plot(df.index + 1, df[c], label=c)
            for c in val_acc_cols:
                plt.plot(df.index + 1, df[c], label=c)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training vs Validation Accuracy")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(out_acc, dpi=150)
            plt.close()

        # Loss plot
        if loss_cols or val_loss_cols:
            plt.figure(figsize=(7, 5))
            for c in loss_cols:
                plt.plot(df.index + 1, df[c], label=c)
            for c in val_loss_cols:
                plt.plot(df.index + 1, df[c], label=c)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training vs Validation Loss")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(out_loss, dpi=150)
            plt.close()

        # Combined (if any)
        if (acc_cols or val_acc_cols) or (loss_cols or val_loss_cols):
            fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
            if acc_cols or val_acc_cols:
                for c in acc_cols:
                    axes[0].plot(df.index + 1, df[c], label=c)
                for c in val_acc_cols:
                    axes[0].plot(df.index + 1, df[c], label=c)
                axes[0].set_ylabel("Accuracy")
                axes[0].set_title("Accuracy")
                axes[0].legend()
                axes[0].grid(True, linestyle="--", alpha=0.4)
            if loss_cols or val_loss_cols:
                for c in loss_cols:
                    axes[1].plot(df.index + 1, df[c], label=c)
                for c in val_loss_cols:
                    axes[1].plot(df.index + 1, df[c], label=c)
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Loss")
                axes[1].set_title("Loss")
                axes[1].legend()
                axes[1].grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            fig.savefig(out_both, dpi=150)
            plt.close(fig)

        return True
    except Exception:
        return False


def plot_eval(json_path: pathlib.Path = EVAL_JSON_PATH,
              out_path: pathlib.Path = PLOT_EVAL_PATH) -> bool:
    """
    Plot evaluation metrics from a JSON file if present.
    Expected format, e.g.: {"accuracy": 0.93, "loss": 0.24, "top5": 0.98}
    """
    try:
        if not json_path.exists():
            return False
        with open(json_path, "r") as f:
            metrics = json.load(f)
        if not isinstance(metrics, dict) or not metrics:
            return False

        keys = list(metrics.keys())
        vals = [float(metrics[k]) for k in keys]

        plt.figure(figsize=(7, 5))
        plt.bar(keys, vals)
        plt.title("Evaluation Metrics")
        plt.ylabel("Value")
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return True
    except Exception:
        return False


def maybe_plot_after_run() -> None:
    """Manually trigger plotting (safe to call even if no files exist)."""
    _ = plot_history()
    _ = plot_eval()


# Auto-run at interpreter exit (best-effort; safe if files absent)
@atexit.register
def _auto_plot_on_exit():
    try:
        maybe_plot_after_run()
    except Exception:
        pass


# ======================================================================================
# Utility summary
# ======================================================================================
def summary() -> str:
    lines = [
        f"PROJECT_ROOT     : {PROJECT_ROOT}",
        f"DATA_ROOT        : {DATA_ROOT}",
        f"  TRAIN_DIR      : {TRAIN_DIR}",
        f"  TEST_DIR       : {TEST_DIR}",
        f"ARTIFACTS        : {ARTIFACTS}",
        f"BEST_KERAS       : {BEST_KERAS}",
        f"SAVEDMODEL_DIR   : {SAVEDMODEL_DIR} (export={EXPORT_SAVEDMODEL})",
        f"HISTORY_CSV      : {HISTORY_CSV}",
        f"CHECKPOINTS_DIR  : {CHECKPOINTS_DIR}",
        f"TENSORBOARD_LOGS : {TENSORBOARD_LOGDIR}",
        f"CLASS_NAMES_JSON : {CLASS_NAMES_JSON}",
        f"IMG_SIZE (HxWxC) : {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}",
        f"BATCH_SIZE       : {BATCH_SIZE}",
        f"EVAL_BATCH_SIZE  : {EVAL_BATCH_SIZE}",
        f"EPOCHS           : {EPOCHS}",
        f"USE_AUGMENTATION : {USE_AUGMENTATION}",
        f"VAL_SPLIT        : {VALIDATION_SPLIT}",
        f"PAD_TO_SQUARE    : {PAD_TO_SQUARE} (color={PAD_COLOR}, interp={RESIZE_INTERP})",
        f"RUN_TS           : {RUN_TS}",
        f"OMP_THREADS      : {OMP_THREADS}",
        f"TF_INTRAOP       : {TF_INTRAOP_THREADS}",
        f"TF_INTEROP       : {TF_INTEROP_THREADS}",
        f"BACKBONE         : {BACKBONE}",
        f"PRETRAINED       : {PRETRAINED}",
        f"FREEZE           : {FREEZE} (unfreeze last {FROZEN_LAYERS} layers)",
        f"GLOBAL_POOL      : {GLOBAL_POOL}",
        f"DROPOUT          : {DROPOUT}",
        f"LEARNING_RATE    : {LEARNING_RATE}",
    ]
    return "\n".join(lines)