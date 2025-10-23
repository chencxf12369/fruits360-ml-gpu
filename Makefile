# Cross-platform Makefile (macOS / Linux / Windows via Git Bash or PowerShell)
# Uses only relative paths; no absolutes.

# Use bash when available (macOS/Linux/Git Bash). On pure Windows without bash,
# we’ll try PowerShell in setup fallback.
SHELL := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# -------------------------
# Platform detection
# -------------------------
ifeq ($(OS),Windows_NT)
  IS_WINDOWS := 1
else
  IS_WINDOWS := 0
endif

# -------------------------
# Paths & variables (relative)
# -------------------------
PKG     := fruits360
VENV    := .venv
ART     := artifacts

# Python + venv activation per platform
ifeq ($(IS_WINDOWS),1)
  PY      := python
  ACT     := $(VENV)/Scripts/activate
  POWERSHELL := powershell
  PWSH     := pwsh
else
  PY      := python3
  ACT     := $(VENV)/bin/activate
endif

# Optional setup scripts (relative)
SETUP_SH  := scripts/setup_env.sh
SETUP_PS1 := scripts/setup_env.ps1

.PHONY: help setup install train eval plot clean tabs

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  setup     - Create venv & install deps (uses scripts/setup_env.sh or .ps1 if present)"
	@echo "  install   - Install package in editable mode inside venv"
	@echo "  train     - Run training (python -m $(PKG).train)"
	@echo "  eval      - Run evaluation (python -m $(PKG).eval)"
	@echo "  plot      - Regenerate plots via fruits360.config"
	@echo "  clean     - Remove caches and artifacts"
	@echo "  tabs      - Check that recipe lines start with a TAB"

# ---------------------------------------------------------------
# setup: prefer your scripts/setup_env.sh (macOS/Linux/Git Bash),
#        else use scripts/setup_env.ps1 (Windows PowerShell),
#        else minimal venv + editable install as a fallback.
# ---------------------------------------------------------------
setup:
	@if [ -f "$(SETUP_SH)" ]; then \
		echo "[setup] Running $(SETUP_SH) ..."; \
		bash "$(SETUP_SH)"; \
	elif [ "$(IS_WINDOWS)" = "1" ] && [ -f "$(SETUP_PS1)" ]; then \
		echo "[setup] Running $(SETUP_PS1) via PowerShell ..."; \
		if command -v $(PWSH) >/dev/null 2>&1; then \
			$(PWSH) -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$(SETUP_PS1)"; \
		else \
			$(POWERSHELL) -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$(SETUP_PS1)"; \
		fi; \
	else \
		echo "[setup] No setup script found. Creating venv and installing basics..."; \
		$(PY) -m venv "$(VENV)"; \
		. "$(ACT)"; \
		pip install -U pip setuptools wheel; \
		pip install -e .; \
	fi

install:
	@. "$(ACT)"; \
	pip install -e .

train:
	@. "$(ACT)"; \
	$(PY) -m $(PKG).train

eval:
	@. "$(ACT)"; \
	$(PY) -m $(PKG).eval

# Safer one-liner (avoids heredoc/TAB pitfalls)
plot:
	@. "$(ACT)"; \
	$(PY) -c "from fruits360 import config; config.maybe_plot_after_run(); print('[plot] regenerated plots under:', config.ARTIFACTS)"

clean:
	@# pyc / __pycache__
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@# artifacts (relative)
	@rm -rf "$(ART)"/tb_logs "$(ART)"/checkpoints 2>/dev/null || true
	@rm -f  "$(ART)"/*.png "$(ART)"/*.json 2>/dev/null || true
	@echo "[clean] done"

# Quick check that recipe lines begin with TABs (not spaces)
tabs:
	@awk ' \
	  /^[[:alnum:]_.-]+:/{inrule=1;next} \
	  inrule{ if($$0 ~ /^\t/) {tabs++} else if($$0 ~ /^[[:space:]]*$$/) {} else {notabs++} } \
	  /^$$/{inrule=0} \
	  END{ printf("Recipe lines with TAB: %d; non-TAB recipe lines: %d\n", tabs, notabs) }' Makefile
