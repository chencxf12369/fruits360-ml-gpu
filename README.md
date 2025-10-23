Fruits360 –Image Classification on macOS (CPU + GPU)

A TensorFlow / Keras image-classification project for fruit detection,
fully tested on
macOS Sequoia (M1 Max ) with Python 3.11 and GPU (Metal) execution.

----------

Project Structure:

fruits360-ml-gpu/
│
├── src/fruits360/
│   ├── __init__.py         ← Marks directory as a Python package
│   ├── config.py           ← Central configuration (paths, constants, parameters)
│   ├── data.py             ← Builds tf.data pipeline (load, augment, prefetch, cache)
│   ├── model.py            ← Defines MobileNetV2 base + classifier head
│   ├── train.py            ← Main training entrypoint (compile, fit, save best model)
│   ├── eval.py             ← Post-training evaluation and metrics computation
│   ├── infer.py            ← Inference script for new or unseen images
│   ├── plot.py             ← Generates accuracy/loss charts, confusion matrices
│   └── utils.py            ← Helper utilities (summary, logging, checkpoint paths)
│
├── scripts/
│   ├── setup_env.sh        ← Creates virtual environment & installs dependencies
│   ├── snapshot_baseline.sh← Generates Git baseline tag with timestamp
│   └── (optional future scripts) 
│
├── artifacts/              ← Stores checkpoints, training history, exported models
├── requirements.txt         ← Python dependencies list
├── Makefile                 ← Shortcut commands for setup, train, evaluate, clean
├── README.md                ← Complete documentation and usage guide
└── LICENSE                  ← Github Repository license

----------
```
#Archive Tips
zip -r ml-gpu.zip ml-gpu -x 'ml-gpu/.venv/*' 'ml-gpu/artifacts-*' 'ml-gpu/src/fruits360/__pycache*' 'ml-gpu/src/fruits360.egg*'
tar -czf ml-gpu.tar.gz \
  --exclude='ml-gpu/.venv' \
  --exclude='ml-gpu/artifacts-*' \
  --exclude='ml-gpu/src/fruits360/__pycache__*' \
  --exclude='ml-gpu/src/fruits360.egg*' \
  ml-gpu
```
----------


0. Prerequisites

  ------------------------------- -------------- --------------------------
  Component                       Version        Install Hint
  macOS Sequoia (Apple-Silicon)   –              Built-in
  Homebrew Python                 3.11 (arm64)   brew install python@3.11
  Git (optional)                  latest         brew install git
  Command Line Tools              –              xcode-select --install
  ------------------------------- -------------- --------------------------


----------
Environment Setup
Optimized for Mac Studio M1 Max (64 GB). One environment only: **GPU (Metal)**.
- Mixed precision enabled
- MobileNetV2 @ 224 with BN frozen + tail fine-tuning
- Hardcoded thread counts (P-cores): OMP/INTRA=8, INTER=2
- Fast tf.data (cache/shuffle/prefetch)

## 1) Setup
```bash
bash scripts/setup_env.sh
#By defayult, it runs GPU but also automatically fall back to CPU if no GPU detected.
#Alternatively, execute the following in the project environment.
export FRUITS360_CPU_ONLY=0 # or export FRUITS360_FORCE_CPU=0  (legacy compatible) ##Run  everything with GPU.
export FRUITS360_CPU_ONLY=1 # or export FRUITS360_FORCE_CPU=1  (legacy compatible) ##For CPU Run only
export FRUITS360_BATCH_SIZE=16 ##For CPU Run only, since the Image_Size is 224X224X3, use BATCH_SIZE=16 instead of 64(GPU use) for better  performance.


##2) Download the Dataset
Use script (or manual copy):
bash scripts/download_data.sh

Dataset is expected under:
~/data/Fruit-Images-Dataset/Training
~/data/Fruit-Images-Dataset/Test
```
## 2) Train the Model
```bash
cd ~/Documents/ml-gpu
source .venv/bin/activate

##full cleanup if necessary
##
##rm -rf src/fruits360.egg-info
##pip uninstall fruits360 -y
##deactivate
cd ~/Documents/ml-gpu
source .venv/bin/activate
pip install -e .

python -m fruits360.train

cd ~/Documents/ml-gpu
pip install -e .

python -m fruits360.train

The console prints detected devices:

    TF: 2.15.0
    GPUs visible: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    CPUs visible: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]

Artifacts land in ./artifacts/:

fruits360_best.keras

fruits360_savedmodel/

train_log.csv
```

## 3) Evaluate on official Test split
```bash
python -m fruits360.eval

```
## 4) Inference
```bash
python -m fruits360.infer --image "$HOME/data/Fruit-Images-Dataset/Test/Apple Golden 2/321_100.jpg"

Notes

CPU-only enforced by env and tf.config.set_visible_devices([], "GPU").

Threading defaults: OMP_NUM_THREADS=18, TF_NUM_INTRAOP_THREADS=18, TF_NUM_INTEROP_THREADS=2.
Tune per workload; for heavier convs, INTEROP=1–2 and INTRA=cores are good starting points.

Reproduce env: make freeze then reinstall with pip install -r requirements.lock.txt


```
## 5) Summary of Workflow
```bash

    # 1.  Create and initialize environment
    bash scripts/setup_env.sh
    export FRUITS360_CPU_ONLY=0 # or export FRUITS360_FORCE_CPU=0  (legacy compatible) ##Run  everything with GPU.
    export FRUITS360_CPU_ONLY=1 # or export FRUITS360_FORCE_CPU=1  (legacy compatible) ##Run  everything with CPU only.
    # 2.  Fetch dataset
    bash scripts/download_data.sh

    # 3.  Activate environment
    source ~/.venvs/bin/activate

    # 4.  Install project
    cd ~/Documents/ml-gpu
    pip install -e .

    # 5.  Train
    python -m fruits360.train

    #6.  Evaluation
    python -m fruits360.eval

    #7. Inference
    python -m fruits360.infer --image "$HOME/data/Fruit-Images-Dataset/Test/Apple Golden 2/321_100.jpg"

```



## 6) Troubleshooting
```bash
  --------------------------------------------------- ---------------------------------------------------------------------------------
  Symptom                                             Cause / Fix
  ModuleNotFoundError: No module named 'tensorflow'   Ensure you’e inside the correct venv (which python) and installed requirements.
  Metal device not found                              Run native arm64 Terminal and reinstall GPU env.
  Slow training on M1/M2                              Confirm you’e using legacy.Adam optimizer (already fixed in model.py).
  Logs mix between runs                               Each venv has its own artifact root (artifacts-cpu vs artifacts-gpu).
  --------------------------------------------------- ---------------------------------------------------------------------------------
```


## 7) Revision History:
```bash
#Revision:
#0.0.1 Raw script for basic scripts to run TensorFlow  with cpu only.
#0.1.0 Adjust threads to optimize the host cpu via /ml/src/fruits360/config.py

#0.1.2 TensorFlow Log quite TF_CPP_MIN_LOG_LEVEL and TF_ENABLE_ONEDNN_OPTS  math leverage to compare performance via /ml/scripts/setup_env.sh.
#0.1.0 Introduce framework of proper project structure layout.
#1.1.0 introduce pyproject.toml (manifest of project metadata),  including definition of python src directory for the  remmedy of execution error of train and evaluate module not found.
#1.2.0 add frozen config via FREEZE_BASE
#1.3.1 add transfer learning  MobileNetV2(ImageNet weights)
#1.3.1 Metrics include accuracy(+ optional top5)
#1.3.1 Checkpoints: best .keras + SavedModel
#2.0.1 Migrate from Ubuntu 25.04 to run on Mac OS
#2.1.1 create additional environment for GPU.
#2.1.1 Log separation between two environments(CPU only and GPU).
#2.2.1 Log plot addition.
#2.2.2 add fallback of setup script failure due to pip has project info without proper metadata, remove stale folder from site-packages.
```

## 8) References
```bash
• TensorFlow Metal Plugin Guide
• Keras API Docs
• Fruits 360 Dataset (Kaggle)
```
## 9)Git backup with Tag creation and push to Github.
```bash
bash scripts/snapshot_baseline.sh
git push origin main --tags
git status
```
## 10) Restore or clone from GitHub
```bash
git clone https://github.com/<you>/fruits360-ml-gpu.git
cd fruits360-ml-gpu
git checkout baseline-[revision]
python3 -m venv .venv && source .venv/bin/activate
pip install -r artifacts/requirements.txt
python -m fruits360.infer --image ...
```
