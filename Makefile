SHELL := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY      := python3
PKG     := fruits360
VENV    := .venv
ACT     := $(VENV)/bin/activate
ART     := artifacts

.PHONY: help setup install train eval plot clean tabs

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  setup     - Run scripts/setup_env.sh (create venv, install TF)"
	@echo "  install   - pip install -e . (inside venv)"
	@echo "  train     - Train model"
	@echo "  eval      - Evaluate model"
	@echo "  plot      - Regenerate plots"
	@echo "  clean     - Remove caches/artifacts"
	@echo "  tabs      - Check TAB indentation"

# ---------------------------------------------------------------
# Setup: delegate to scripts/setup_env.sh
# ---------------------------------------------------------------
setup:
	@if [ -x "scripts/setup_env.sh" ]; then \
		echo "[setup] Running scripts/setup_env.sh ..."; \
		bash scripts/setup_env.sh; \
	else \
		echo "[setup] scripts/setup_env.sh not found or not executable."; \
		echo "[setup] Creating venv manually..."; \
		$(PY) -m venv $(VENV); \
		source $(ACT); \
		pip install -U pip setuptools wheel; \
	fi

install:
	@source $(ACT); \
	pip install -e .

train:
	@source $(ACT); \
	$(PY) -m $(PKG).train

eval:
	@source $(ACT); \
	$(PY) -m $(PKG).eval

plot:
	@source $(ACT); \
	$(PY) -c "from fruits360 import config; config.maybe_plot_after_run(); print('[plot] regenerated plots under:', config.ARTIFACTS)"

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf $(ART)/*.png $(ART)/*.json $(ART)/tb_logs $(ART)/checkpoints 2>/dev/null || true
	@echo "[clean] done"

tabs:
	@awk ' \
	  /^[[:alnum:]_.-]+:/{inrule=1;next} \
	  inrule{ if($$0 ~ /^\t/) {tabs++} else if($$0 ~ /^[[:space:]]*$$/) {} else {notabs++} } \
	  /^$$/{inrule=0} \
	  END{ printf("Recipe lines with TAB: %d; non-TAB recipe lines: %d\n", tabs, notabs) }' Makefile
