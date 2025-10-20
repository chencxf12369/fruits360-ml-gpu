
# Makefile for fruits360 (TAB-safe)
PY      := python
PKG     := fruits360
VENV    := .venv
ACT     := $(VENV)/bin/activate
ART     := artifacts

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  setup     - Create venv, install deps"
	@echo "  install   - pip install -e ."
	@echo "  train     - python -m fruits360.train"
	@echo "  eval      - python -m fruits360.eval"
	@echo "  infer     - make infer IMAGE=/path/img.jpg"
	@echo "  plot      - regenerate plots from artifacts"
	@echo "  clean     - remove caches and artifacts/*"
	@echo "  tabs      - count recipe lines starting with TAB"

setup:
	@if [ -f "scripts/setup_env.sh" ]; then \\
		bash scripts/setup_env.sh; \\
	else \\
		echo "[setup] no scripts/setup_env.sh -> falling back to make install"; \\
		$(MAKE) install; \\
	fi

install:
	$(PY) -m pip install -e .

train:
	$(PY) -m $(PKG).train

eval:
	$(PY) -m $(PKG).eval

infer:
	@if [ -z "$(IMAGE)" ]; then \\
		echo "Usage: make infer IMAGE=/absolute/path/to/file.jpg"; \\
		exit 2; \\
	fi
	$(PY) -m $(PKG).infer --image "$(IMAGE)"

plot:
	$(PY) - <<'EOF'
from fruits360 import config
config.maybe_plot_after_run()
print("[plot] regenerated plots under:", config.ARTIFACTS)
EOF

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf $(ART)/*.png $(ART)/*.json $(ART)/tb_logs $(ART)/checkpoints 2>/dev/null || true
	@echo "[clean] done"

tabs:
	@echo "Counting recipe lines that begin with a real TAB..."
	@awk '/^[[:alnum:]_.-]+:/{rule=1;next} rule{if($$0 ~ /^\	/) c++; if($$0 ~ /^[^#[:space:]]/ || $$0 ~ /^$$/) rule=0} END{print c, "recipe lines start with a TAB"}' Makefile
