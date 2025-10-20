#!/usr/bin/env bash
set -euo pipefail

# --- config ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="$ROOT_DIR/artifacts"
TS="$(date +%Y%m%d-%H%M%S)"
TAG="baseline-${TS}"
MANIFEST="$ART_DIR/baseline_${TS}.manifest.json"

echo "[snap] Project root: $ROOT_DIR"
cd "$ROOT_DIR"

# 1) Verify venv + capture Python & TF/Keras versions
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[snap] WARNING: not in a virtualenv. Proceeding anyway."
else
  echo "[snap] Using venv: $VIRTUAL_ENV"
fi

mkdir -p "$ART_DIR"

echo "[snap] Capturing pip freeze -> artifacts/requirements.txt"
python - <<'PY' > artifacts/requirements.txt
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False)
PY

# 2) Capture fruits360 config summary (so we remember image size, batch, etc.)
echo "[snap] Capturing config summary -> artifacts/config-summary.txt"
python - <<'PY' > artifacts/config-summary.txt
from fruits360 import config
print(config.summary())
PY

# 3) Persist class names (if not already there)
echo "[snap] Capturing class names -> artifacts/class_names.json"
python - <<'PY'
import json, pathlib
from fruits360 import config
p = pathlib.Path(config.ARTIFACTS) / "class_names.json"
if not p.exists():
    # prefer TRAIN_DIR ordering
    root = pathlib.Path(config.TRAIN_DIR)
    if root.exists():
        names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    else:
        names = []
    p.write_text(json.dumps(names, indent=2))
print(str(p))
PY

# 4) Hash key artifacts for reproducibility
echo "[snap] Computing SHA256 of key artifacts -> artifacts/hashes_${TS}.txt"
( cd "$ART_DIR" && \
  shasum -a 256 \
    fruits360_best.keras \
    2>/dev/null || true
  shasum -a 256 \
    $(find fruits360_savedmodel -type f 2>/dev/null | sort) \
    2>/dev/null || true
) > "$ART_DIR/hashes_${TS}.txt" || true

# 5) Build a manifest with env + paths
echo "[snap] Writing manifest -> $MANIFEST"
python - <<PY
import json, os, platform, pathlib, time
from fruits360 import config
m = {
  "timestamp": "${TS}",
  "python": platform.python_version(),
  "platform": platform.platform(),
  "env": {
    k: os.environ[k] for k in sorted(os.environ)
    if k.startswith("FRUITS360_") or k in ("OMP_NUM_THREADS","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS")
  },
  "paths": {
    "project_root": str(pathlib.Path("${ROOT_DIR}").resolve()),
    "artifacts": str(pathlib.Path("${ART_DIR}").resolve()),
    "best_keras": str(config.BEST_KERAS),
    "savedmodel": str(config.SAVEDMODEL_DIR),
    "history_csv": str(config.HISTORY_CSV),
    "tb_logdir": str(config.TENSORBOARD_LOGDIR),
  }
}
path = pathlib.Path("${MANIFEST}")
path.write_text(json.dumps(m, indent=2))
print(path)
PY

# 6) Git commit + tag
echo "[snap] Creating git commit & tag"
git add -A
git commit -m "Baseline snapshot ${TS}: code+env+artifacts captured" || true
git tag -a "${TAG}" -m "Baseline snapshot ${TS}"

echo
echo "[snap] Done."
echo "       Tag      : ${TAG}"
echo "       Manifest : ${MANIFEST}"
echo
echo "Rollback to this snapshot later with:"
echo "  git checkout ${TAG}"
