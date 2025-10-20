# src/fruits360/utils.py
from __future__ import annotations

import os
import platform
import tensorflow as tf
from . import config


def _print_devices(prefix: str = "[device]") -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        cpus = tf.config.list_physical_devices("CPU")
    except Exception:
        gpus, cpus = [], []
    print(f"{prefix} GPUs visible:", gpus)
    print(f"{prefix} CPUs visible:", cpus)


def force_cpu_only(verbose: bool = True) -> None:
    """
    Enforce CPU-only execution for TensorFlow:
      - Hides all GPUs from TF
      - Sets CUDA/HIP visibility vars
      - Keeps logs quieter
    """
    # Hide GPUs from TF runtime
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    # Also tell other stacks to stay off GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    if verbose:
        print("[device] CPU-only mode enabled")
        _print_devices()


def tune_threads(verbose: bool = True) -> None:
    """
    Apply threading env defaults from config and try TF threading APIs.
    """
    # Environment variables (picked up early by many libs)
    os.environ.setdefault("OMP_NUM_THREADS",        str(config.OMP_THREADS))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(config.TF_INTRAOP_THREADS))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(config.TF_INTEROP_THREADS))

    # Best-effort: also ask TF runtime
    try:
        tf.config.threading.set_intra_op_parallelism_threads(config.TF_INTRAOP_THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(config.TF_INTEROP_THREADS)
    except Exception:
        # Harmless if the runtime is already initialized.
        pass

    if verbose:
        print(
            "Threading:",
            "OMP_NUM_THREADS=", os.environ.get("OMP_NUM_THREADS"),
            "TF_NUM_INTRAOP_THREADS=", os.environ.get("TF_NUM_INTRAOP_THREADS"),
            "TF_NUM_INTEROP_THREADS=", os.environ.get("TF_NUM_INTEROP_THREADS"),
        )


def ensure_device(prefer_gpu: bool = True, verbose: bool = True) -> None:
    """
    One call to set up compute device + threads for train/eval/infer.

    Honors the following env flags:
      - FRUITS360_CPU_ONLY=1  -> force CPU (always)
      - FRUITS360_FORCE_CPU=1 -> force CPU (legacy / your previous behavior)

    Apple Silicon note:
      We do *not* hide the Metal GPU unless you explicitly set one of the flags above.
    """
    # Hard overrides
    if os.environ.get("FRUITS360_CPU_ONLY", "0") == "1" or os.environ.get("FRUITS360_FORCE_CPU", "0") == "1":
        force_cpu_only(verbose=verbose)
        tune_threads(verbose=verbose)
        return

    mac_arm = (platform.system() == "Darwin" and platform.machine() == "arm64")

    # Default behavior: prefer GPU if present; otherwise run on CPU.
    gpus = []
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        pass

    if gpus:
        if verbose:
            if mac_arm:
                print("[device] Apple Silicon detected -> GPU (Metal) enabled")
            else:
                print("[device] GPU detected -> using GPU")
            _print_devices()
        # Nothing to configure—TF will use the visible GPU.
    else:
        if verbose:
            print("[device] No GPU available -> running on CPU")
            _print_devices()

    # Always tune threads afterwards
    tune_threads(verbose=verbose)