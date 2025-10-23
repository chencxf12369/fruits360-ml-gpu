# =====================================================
# Cross-platform Makefile for fruits360-ml-gpu project
# Supports macOS, Linux, and Windows (Git Bash / PowerShell)
# Includes GPU/CPU portability and GitHub-only dataset fetch
# =====================================================

# --- OS and shell detection ---
ifeq ($(OS),Windows_NT)
  IS_WINDOWS := 1
else
  IS_WINDOWS := 0
endif

BASH_PATH := $(shell command -v bash 2>/dev/null || which bash 2>/dev/null)
ifeq ($(strip $(BASH_PATH)),)
  HAS_BASH := 0
else
  HAS_BASH := 1
endif

ifeq ($(HAS_BASH),1)
  SHELL := $(BASH_PATH)
else
  ifeq ($(IS_WINDOWS),1)
    SHELL := cmd.exe
  else
    SHELL := /bin/sh
  endif
endif

ifneq ($(SHELL),cmd.exe)
  .ONESHELL:
  .SHELLFLAGS := -eu -o pipefail -c
endif

# --- Core paths and variables ---
PKG     := fruits360
VENV    := .venv
ART     := artifacts
DATA_DIR_REPO := Fruit-Images-Dataset

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

SETUP_SH  := scripts/setup_env.sh
SETUP_PS1 := scripts/setup_env.ps1

.PHONY: help setup install train eval plot clean tabs \
        dataset dataset-github dataset-check dataset-where dataset-clean

.DEFAULT_GOAL := help

# =====================================================
# Help
# =====================================================
help:
	@echo "Targets:"
	@echo "  setup           - Create venv & install deps (auto-selects scripts/setup_env.sh or .ps1)"
	@echo "  install         - pip install -e . (inside venv)"
	@echo "  train           - python -m $(PKG).train"
	@echo "  eval            - python -m $(PKG).eval"
	@echo "  plot            - regenerate plots"
	@echo "  clean           - remove caches and artifacts"
	@echo "  tabs            - check TAB indentation in recipes"
	@echo ""
	@echo "Dataset targets (GitHub only):"
	@echo "  dataset         - Install or update dataset from GitHub"
	@echo "  dataset-check   - Verify dataset folders exist"
	@echo "  dataset-where   - Print dataset path used by code"
	@echo "  dataset-clean   - Remove local dataset copy"
	@echo ""
	@echo "Detected: IS_WINDOWS=$(IS_WINDOWS) HAS_BASH=$(HAS_BASH) SHELL=$(SHELL)"

# =====================================================
# Environment setup
# =====================================================
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

# =====================================================
# Project execution
# =====================================================
install:
ifeq ($(IS_WINDOWS),1)
	@powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '$(ACT_PS1)'; pip install -e .;"
else
	@. "$(ACT_BASH)"; \
	pip install -e .
endif

train:
ifeq ($(IS_WINDOWS),1)
	@powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '$(ACT_PS1)'; $(PY) -m $(PKG).train"
else
	@. "$(ACT_BASH)"; \
	$(PY) -m $(PKG).train
endif

eval:
ifeq ($(IS_WINDOWS),1)
	@powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '$(ACT_PS1)'; $(PY) -m $(PKG).eval"
else
	@. "$(ACT_BASH)"; \
	$(PY) -m $(PKG).eval
endif

plot:
ifeq ($(IS_WINDOWS),1)
	@powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '$(ACT_PS1)'; $(PY) -c \"from fruits360 import config; config.maybe_plot_after_run(); print('[plot] regenerated plots under:', config.ARTIFACTS)\""
else
	@. "$(ACT_BASH)"; \
	$(PY) -c "from fruits360 import config; config.maybe_plot_after_run(); print('[plot] regenerated plots under:', config.ARTIFACTS)"
endif

# =====================================================
# Cleanup and validation
# =====================================================
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

tabs:
	@awk ' \
	  /^[[:alnum:]_.-]+:/{inrule=1;next} \
	  inrule{ if($$0 ~ /^\t/) {tabs++} else if($$0 ~ /^[[:space:]]*$$/) {} else {notabs++} } \
	  /^$$/{inrule=0} \
	  END{ printf("Recipe lines with TAB: %d; non-TAB recipe lines: %d\n", tabs, notabs) }' Makefile

# =====================================================
# Dataset management (GitHub only)
# =====================================================
dataset: dataset-github dataset-check

dataset-github:
	@echo "[dataset] Installing Fruits-360 dataset via GitHub..."
	@if [ -d "$(DATA_DIR_REPO)/.git" ]; then \
	  echo "[dataset] Existing repo found, pulling latest updates..."; \
	  git -C "$(DATA_DIR_REPO)" pull --ff-only || true; \
	elif [ -d "$(DATA_DIR_REPO)/Training" ] && [ -d "$(DATA_DIR_REPO)/Test" ]; then \
	  echo "[dataset] Dataset already exists locally at ./$(DATA_DIR_REPO)"; \
	else \
	  if command -v git >/dev/null 2>&1; then \
	    git clone --depth=1 https://github.com/Horea94/Fruit-Images-Dataset "$(DATA_DIR_REPO)"; \
	  elif command -v curl >/dev/null 2>&1; then \
	    echo "[dataset] Cloning via curl (no git found)..."; \
	    curl -L -o fruit-images-dataset.zip https://codeload.github.com/Horea94/Fruit-Images-Dataset/zip/refs/heads/master; \
	    unzip -q -o fruit-images-dataset.zip; \
	    mv -f Fruit-Images-Dataset-master "$(DATA_DIR_REPO)"; \
	    rm -f fruit-images-dataset.zip; \
	  elif command -v wget >/dev/null 2>&1; then \
	    echo "[dataset] Cloning via wget (no git found)..."; \
	    wget -O fruit-images-dataset.zip https://codeload.github.com/Horea94/Fruit-Images-Dataset/zip/refs/heads/master; \
	    unzip -q -o fruit-images-dataset.zip; \
	    mv -f Fruit-Images-Dataset-master "$(DATA_DIR_REPO)"; \
	    rm -f fruit-images-dataset.zip; \
	  else \
	    echo "[dataset] ERROR: Need git, curl, or wget to fetch dataset."; \
	    exit 1; \
	  fi; \
	fi
	@echo "[dataset] Installed into ./$(DATA_DIR_REPO)"

dataset-check:
	@if [ -d "$(DATA_DIR_REPO)/Training" ] && [ -d "$(DATA_DIR_REPO)/Test" ]; then \
	  echo "[dataset-check] OK: $(DATA_DIR_REPO) has Training/ and Test/"; \
	else \
	  echo "[dataset-check] ERROR: Missing Training/ or Test/ under $(DATA_DIR_REPO)"; \
	  exit 1; \
	fi

dataset-where:
	@echo "[dataset-where] Dataset location: ./$(DATA_DIR_REPO)"

dataset-clean:
	@echo "[dataset-clean] Removing ./$(DATA_DIR_REPO)..."
	@rm -rf "$(DATA_DIR_REPO)"
	@echo "[dataset-clean] Done."
