from __future__ import annotations
import os
import platform
import tensorflow as tf
from . import config

def force_cpu_only(verbose: bool = True) -> None:
    """
    Respect Apple Silicon GPU by default; allow forcing CPU with env.
    Aliases: FRUITS360_FORCE_CPU=1 or FRUITS360_CPU_ONLY=1
    """
    if os.environ.get("FRUITS360_CPU_ONLY", "") in {"1","true","True"}:
        os.environ["FRUITS360_FORCE_CPU"] = "1"

    mac_arm = (platform.system() == "Darwin" and platform.machine() == "arm64")
    if mac_arm and os.environ.get("FRUITS360_FORCE_CPU", "0") != "1":
        if verbose:
            print("Apple Silicon detected -> GPU stays enabled. "
                  "Set FRUITS360_FORCE_CPU=1 (or FRUITS360_CPU_ONLY=1) to force CPU.")
            print("GPUs visible:", tf.config.list_physical_devices("GPU"))
            print("CPUs visible:", tf.config.list_physical_devices("CPU"))
        return

    # CPU-only
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    if verbose:
        print("GPUs visible:", tf.config.list_physical_devices("GPU"))
        print("CPUs visible:", tf.config.list_physical_devices("CPU"))

def ensure_device(prefer_gpu: bool = True) -> None:
    """
    Light-weight device policy: use CPU if FRUITS360_FORCE_CPU/FRUITS360_CPU_ONLY set,
    otherwise prefer GPU if available.
    """
    if os.environ.get("FRUITS360_CPU_ONLY", "") in {"1","true","True"}:
        os.environ["FRUITS360_FORCE_CPU"] = "1"
    if os.environ.get("FRUITS360_FORCE_CPU", "0") == "1":
        force_cpu_only(verbose=True)
        return
    # Else: leave GPU visible if any; nothing to do.
    print("GPUs visible:", tf.config.list_physical_devices("GPU"))

def tune_threads(verbose: bool = True) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(config.OMP_THREADS))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(config.TF_INTRA))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(config.TF_INTER))
    if verbose:
        print("Threading:",
              "OMP_NUM_THREADS=", os.environ.get("OMP_NUM_THREADS"),
              "TF_NUM_INTRAOP_THREADS=", os.environ.get("TF_NUM_INTRAOP_THREADS"),
              "TF_NUM_INTEROP_THREADS=", os.environ.get("TF_NUM_INTEROP_THREADS"))


import random, numpy as np, tensorflow as tf
def setup_seed(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and TensorFlow to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[utils] Random seed set to {seed}")
