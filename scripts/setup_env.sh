#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------
# Setup for Fruits-360 project (macOS M1/M2 GPU or Linux CPU)
# ---------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "[setup] Project root: $PROJECT_ROOT"
echo "[setup] Detected OS=$OS ARCH=$ARCH"

# ---------------------------------------------------------------------
# Python selection
# ---------------------------------------------------------------------
if [[ "$OS" == "Darwin" ]]; then
  # Prefer Homebrew Python 3.11; fall back to python3 if not found.
  if [[ -x "/opt/homebrew/bin/python3.11" ]]; then
    PY="/opt/homebrew/bin/python3.11"
  else
    PY="$(command -v python3 || true)"
  fi
else
  PY="$(command -v python3 || true)"
fi

if [[ -z "${PY:-}" || ! -x "$PY" ]]; then
  echo "ERROR: Python 3 not found."
  echo "On macOS/arm64:  brew install python@3.11"
  exit 1
fi

# ---------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------
echo "[setup] Creating venv (.venv)"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# ---------------------------------------------------------------------
# Package installation
# ---------------------------------------------------------------------
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  echo "[setup] Installing TensorFlow Metal stack for Apple Silicon"
  set +e
  python -m pip install "tensorflow-macos==2.16.1"
  TF_STATUS=$?
  set -e
  if [[ $TF_STATUS -ne 0 ]]; then
    echo "[warn] tensorflow-macos 2.16.1 unavailable – falling back to 2.15.0"
    python -m pip install "tensorflow-macos==2.15.0"
  fi
  python -m pip install "tensorflow-metal==1.1.0"
else
  echo "[setup] Non-macOS → CPU-only install from requirements.txt (if present)"
  if [[ -f requirements.txt ]]; then
    python -m pip install -r requirements.txt
  fi
fi

echo "[setup] Installing common Python packages"
python -m pip install "numpy<2.0" pandas==2.2.3 pillow==10.4.0 \
  matplotlib==3.9.2 scikit-learn==1.5.2 opencv-python-headless==4.10.0.84 tqdm==4.66.5

# ---------------------------------------------------------------------
# Install local package (editable) and verify import
# ---------------------------------------------------------------------
echo "[setup] Installing fruits360 (editable)"
if ! python -m pip install -e . ; then
  echo "[setup] Editable install failed; cleaning stale fruits360 entries and retrying..."
  python - <<'PY'
import sys, sysconfig, pathlib, shutil
sp = pathlib.Path(sysconfig.get_paths()['purelib'])
for d in sp.glob('fruits360*'):
    if d.is_dir():
        print("Removing", d)
        shutil.rmtree(d, ignore_errors=True)
PY
  python -m pip install --force-reinstall --no-deps -e .
fi

# ---------------------------------------------------------------------
# Permanent sys.path fix via .pth file
# ---------------------------------------------------------------------
echo "[setup] Creating .pth file for src/ import path"
python - <<'PY'
import site, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[1]   # .../ml-gpu
site_pkgs = next(p for p in site.getsitepackages() if p.endswith("site-packages"))
pth = pathlib.Path(site_pkgs) / "fruits360_src.pth"
pth.write_text(str(root / "src") + "\n")
print(f"[verify] wrote {pth}")
sys.path.append(str(root / "src"))
import fruits360, importlib, pathlib as _p
print("[verify] import fruits360 OK:", _p.Path(fruits360.__file__).resolve())
importlib.import_module("fruits360.plot")
print("[verify] fruits360.plot import OK")
PY

# ---------------------------------------------------------------------
# Environment variables & cleanup
# ---------------------------------------------------------------------
export TF_CPP_MIN_LOG_LEVEL=1
if [[ "$OS" != "Darwin" ]]; then
  export TF_ENABLE_ONEDNN_OPTS=0
fi
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
ACT=.venv/bin/activate
grep -q 'export PYTHONPATH=' "$ACT" || printf '\n# ensure src on sys.path\nexport PYTHONPATH="%s/src:${PYTHONPATH:-}"\n' "$PROJECT_ROOT" >> "$ACT"

echo "[setup] "
echo "[setup] Activate your env with:"
echo "source \"$PROJECT_ROOT/.venv/bin/activate\""
echo "[setup] Then run:"
echo "python -m fruits360.train"
echo "python -m fruits360.eval"
echo "python -m fruits360.plot"
echo "python -m fruits360.infer --image \"$HOME/data/Fruit-Images-Dataset/Test/Apple Golden 2/321_100.jpg\""
echo "[setup] Environment ready."
