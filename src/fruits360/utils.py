from __future__ import annotations
import os
import platform
import tensorflow as tf
from . import config

def force_cpu_only(verbose: bool = True) -> None:
    """
    On Linux/Windows keep current behavior unless FRUITS360_FORCE_CPU=0.
    On Apple Silicon (Darwin/arm64), DO NOT hide the GPU by default.
    You can still force CPU by setting FRUITS360_FORCE_CPU=1.
    """
    mac_arm = (platform.system() == "Darwin" and platform.machine() == "arm64")
    if mac_arm and os.environ.get("FRUITS360_FORCE_CPU", "0") != "1":
        if verbose:
            print("Apple Silicon detected -> GPU stays enabled. "
                  "Set FRUITS360_FORCE_CPU=1 to force CPU.")
            print("GPUs visible:", tf.config.list_physical_devices("GPU"))
            print("CPUs visible:", tf.config.list_physical_devices("CPU"))
        return

    # original CPU-only behavior
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

def tune_threads(verbose: bool = True) -> None:
    # Good defaults for i9-10980XE
    os.environ.setdefault("OMP_NUM_THREADS", str(config.OMP_THREADS))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(config.TF_INTRA))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(config.TF_INTER))
    if verbose:
        print("Threading:",
              "OMP_NUM_THREADS=", os.environ.get("OMP_NUM_THREADS"),
              "TF_NUM_INTRAOP_THREADS=", os.environ.get("TF_NUM_INTRAOP_THREADS"),
              "TF_NUM_INTEROP_THREADS=", os.environ.get("TF_NUM_INTEROP_THREADS"))

