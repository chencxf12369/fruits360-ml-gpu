# ---------------------------
# Cross-platform, self-detecting Makefile
# Works on: macOS, Linux, Windows (Git Bash or PowerShell)
# Uses relative paths only.
# ---------------------------

# --- OS & shell detection (Make conditionals) ---
ifeq ($(OS),Windows_NT)
  IS_WINDOWS := 1
else
  IS_WINDOWS := 0
endif

# Try to locate bash (works on macOS/Linux/Git Bash). Falls back to sh on Unix.
BASH_PATH := $(shell command -v bash 2>/dev/null || which bash 2>/dev/null)
ifeq ($(strip $(BASH_PATH)),)
  HAS_BASH := 0
else
  HAS_BASH := 1
endif

# Select SHELL for Make. GNU Make requires a single executable path.
# - If bash exists, use it (best for our recipes).
# - Else if on Windows without bash, use cmd.exe (we'll call PowerShell explicitly in recipes).
# - Else use /bin/sh (POSIX sh).
ifeq ($(HAS_BASH),1)
  SHELL := $(BASH_PATH)
else
  ifeq ($(IS_WINDOWS),1)
    SHELL := cmd.exe
  else
    SHELL := /bin/sh
  endif
endif

# Keep recipes strict when we do have a POSIX shell.
ifneq ($(SHELL),cmd.exe)
  .ONESHELL:
  .SHELLFLAGS := -eu -o pipefail -c
endif

# ---------------------------
# Project variables (relative)
# ---------------------------
PKG     := fruits360
VENV    := .venv
ART     := artifacts

# Python & activation paths per platform
ifeq ($(IS_WINDOWS),1)
  PY          := python
  ACT_BASH    := $(VENV)/Scripts/activate
  ACT_PS1     := $(VENV)/Scripts/Activate.ps1
  POWERSHELL  := powershell
  PWSH        := pwsh
else
  PY          := python3
  ACT_BASH    := $(VENV)/bin/activate
endif

# Optional user scripts
SETUP_SH  := scripts/setup_env.sh
SETUP_PS1 := scripts/setup_env.ps1

.PHONY: help setup install train eval plot clean tabs

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  setup   - Create venv & install deps (auto-selects scripts/setup_env.sh or .ps1)"
	@echo "  install - pip install -e . (inside venv)"
	@echo "  train   - python -m $(PKG).train"
	@echo "  eval    - python -m $(PKG).eval"
	@echo "  plot    - regenerate plots"
	@echo "  clean   - remove caches and artifacts"
	@echo "  tabs    - check TAB indentation in recipes"
	@echo ""
	@echo "Detected: IS_WINDOWS=$(IS_WINDOWS) HAS_BASH=$(HAS_BASH) SHELL=$(SHELL)"

# ---------------------------
# setup
# ---------------------------
# Logic:
# 1) If scripts/setup_env.sh exists and we have bash -> run it.
# 2) Else if on Windows and scripts/setup_env.ps1 exists -> run via PowerShell.
# 3) Else create venv + minimal install in a portable way.
setup:
ifeq ($(HAS_BASH),1)
	@if [ -f "$(SETUP_SH)" ]; then \
	  echo "[setup] Running $(SETUP_SH) with bash: $(BASH_PATH)"; \
	  "$(BASH_PATH)" "$(SETUP_SH)"; \
	else \
	  echo "[setup] No $(SETUP_SH). Creating venv and installing basics..."; \
	  $(PY) -m venv "$(VENV)"; \
	  . "$(ACT_BASH)"; \
	  pip install -U pip setuptools wheel; \
	  pip install -e .; \
	fi
else
ifeq ($(IS_WINDOWS),1)
	@if exist "$(SETUP_PS1)" ( \
	  echo [setup] Running $(SETUP_PS1) via PowerShell ... & \
	  ( where $(PWSH) >NUL 2>&1 && $(PWSH) -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$(SETUP_PS1)" ) || \
	  ( $(POWERSHELL) -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$(SETUP_PS1)" ) \
	) else ( \
	  echo [setup] No $(SETUP_PS1). Creating venv and installing basics... & \
	  $(PY) -m venv "$(VENV)" & \
	  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '$(ACT_PS1)'; pip install -U pip setuptools wheel; pip install -e .;" \
	)
else
	@echo "[setup] No bash found. Using /bin/sh fallback..."
	@$(PY) -m venv "$(VENV)"
	@. "$(ACT_BASH)"; \
	pip install -U pip setuptools wheel; \
	pip install -e .
endif
endif

# ---------------------------
# install (editable)
# ---------------------------
install:
ifeq ($(IS_WINDOWS),1)
	@powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '$(ACT_PS1)'; pip install -e .;"
else
	@. "$(ACT_BASH)"; \
	pip install -e .
endif

# ---------------------------
# train
# ---------------------------
train:
ifeq ($(IS_WINDOWS),1)
	@powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '$(ACT_PS1)'; $(PY) -m $(PKG).train"
else
	@. "$(ACT_BASH)"; \
	$(PY) -m $(PKG).train
endif

# ---------------------------
# eval
# ---------------------------
eval:
ifeq ($(IS_WINDOWS),1)
	@powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '$(ACT_PS1)'; $(PY) -m $(PKG).eval"
else
	@. "$(ACT_BASH)"; \
	$(PY) -m $(PKG).eval
endif

# ---------------------------
# plot (avoid heredoc/TAB pitfalls)
# ---------------------------
plot:
ifeq ($(IS_WINDOWS),1)
	@powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '$(ACT_PS1)'; $(PY) -c \"from fruits360 import config; config.maybe_plot_after_run(); print('[plot] regenerated plots under:', config.ARTIFACTS)\""
else
	@. "$(ACT_BASH)"; \
	$(PY) -c "from fruits360 import config; config.maybe_plot_after_run(); print('[plot] regenerated plots under:', config.ARTIFACTS)"
endif

# ---------------------------
# clean
# ---------------------------
clean:
ifeq ($(IS_WINDOWS),1)
	@powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command " \
	  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Get-ChildItem -Recurse -Directory -Filter __pycache__).FullName; \
	  Get-ChildItem -Recurse -Filter *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue; \
	  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '$(ART)\tb_logs','$(ART)\checkpoints'; \
	  Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path '$(ART)' '*.png'),(Join-Path '$(ART)' '*.json'); \
	  Write-Host '[clean] done' \
	"
else
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf "$(ART)"/tb_logs "$(ART)"/checkpoints 2>/dev/null || true
	@rm -f  "$(ART)"/*.png "$(ART)"/*.json 2>/dev/null || true
	@echo "[clean] done"
endif

# ---------------------------
# tabs check
# ---------------------------
tabs:
	@awk ' \
	  /^[[:alnum:]_.-]+:/{inrule=1;next} \
	  inrule{ if($$0 ~ /^\t/) {tabs++} else if($$0 ~ /^[[:space:]]*$$/) {} else {notabs++} } \
	  /^$$/{inrule=0} \
	  END{ printf("Recipe lines with TAB: %d; non-TAB recipe lines: %d\n", tabs, notabs) }' Makefile
