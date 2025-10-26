###
A TensorFlow / Keras image-classification project for fruit detection,
fully tested on
macOS Sequoia (M1 Max ) with Python 3.11 and GPU (Metal) execution.

----------
## 📁 Project Structure
----------
```
fruits360-ml-gpu/
│
├── src/fruits360/
│   ├── __init__.py          ← Marks directory as a Python package
│   ├── config.py            ← Central configuration (paths, constants, parameters)
│   ├── data.py              ← tf.data pipeline (load, resize/pad, augment, cache, prefetch)
│   ├── model.py             ← MobileNetV2 backbone + classifier head
│   ├── train.py             ← Training entrypoint (compile, fit, callbacks, save best model)
│   ├── eval.py              ← Post-training evaluation on test/val sets
│   ├── infer.py             ← Inference for unseen images
│   ├── plot.py              ← Accuracy/loss plots, confusion matrix generation
│   └── utils.py             ← Helpers (seeding, logging, paths, utilities)
│
├── scripts/
│   ├── setup_env.sh         ← Creates venv & installs dependencies (macOS/Linux)
│   ├── setup_env.ps1        ← PowerShell setup script (Windows)
│   ├── snapshot_baseline.sh ← Creates Git baseline tag with timestamp
│   └── (optional future scripts)
│
├── artifacts/               ← Model checkpoints, history, TensorBoard logs
├── requirements.txt          ← Minimal dependency list for compatibility
├── pyproject.toml            ← Modern packaging config (PEP 621, setuptools src-layout)
├── Makefile                  ← Shortcut commands: setup, train, eval, plot, clean
├── README.md                 ← Full documentation and usage guide
└── LICENSE                   ← Repository license

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


## 0) Prerequisites
```
Ignore this if it's not Mac OS environment.

  ------------------------------- -------------- --------------------------
  Component                       Version        Install Hint
  macOS Sequoia (Apple-Silicon)   –              Built-in
  Homebrew Python                 3.11 (arm64)   brew install python@3.11
  Git (optional)                  latest         brew install git
  Command Line Tools              –              xcode-select --install
  ------------------------------- -------------- --------------------------
```
----------
## 1) Setup
```bash
bash scripts/setup_env.sh
#By defayult, it runs GPU but also automatically fall back to CPU if no GPU detected.
#Alternatively, execute the following in the project environment for manual tweak in relation to performance finetune.
export FRUITS360_CPU_ONLY=0 # or export FRUITS360_FORCE_CPU=0  (legacy compatible) ##Run  everything with GPU.
export FRUITS360_CPU_ONLY=1 # or export FRUITS360_FORCE_CPU=1  (legacy compatible) ##For CPU Run only

Other Variable Options:
    export FRUITS360_BATCH_SIZE=8 ##For CPU Run only, since the Image_Size is 224X224X3, use BATCH_SIZE=8 instead of 64(GPU use) for better  performance.
    export FRUITS360_OMP_THREADS=8
    export FRUITS360_TF_INTRAOP_THREADS=8
    export FRUITS360_TF_INTEROP_THREADS=2
    # smaller per-step memory
    export FRUITS360_BATCH_SIZE=8
    #shrink shuffle buffer (defaults to BATCH*64 which can be big)
    export FRUITS360_SHUFFLE_BUFFER=2048
    # (optional) if you suspect oneDNN kernel variants chewing RAM/threads
    export TF_ENABLE_ONEDNN_OPTS=0
    # Disable CACHE prefetch for memory exhausting issue, works with TF_ENABLE_ONEDNN_OPTS=1 
    export FRUITS360_CACHE=0
    #run with lower OS priority so the desktop stays responsive
    nice -n 10 python -m fruits360.train
    #disbale PAD globally for traing, test and eva.
    export FRUITS360_PAD_TO_SQUARE=0
    #Temporarily make scalling control to maximum or lower based on the percentage number(1.0=100%,0.5=50%,default is 0.75=75%).
    FRUITS360_THREAD_SCALE=1.0

```

## 2) Download the Dataset
```
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

## 4) Evaluate on official Test split
```bash
python -m fruits360.eval

```
## 5) Inference
```bash
python -m fruits360.infer --image "$HOME/data/Fruit-Images-Dataset/Test/Apple Golden 2/321_100.jpg"

Notes

CPU-only enforced by env and tf.config.set_visible_devices([], "GPU").

Threading defaults: OMP_NUM_THREADS=18, TF_NUM_INTRAOP_THREADS=18, TF_NUM_INTEROP_THREADS=2.
Tune per workload; for heavier convs, INTEROP=1–2 and INTRA=cores are good starting points.

Reproduce env: make freeze then reinstall with pip install -r requirements.lock.txt


```
## 6) Summary of Workflow
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

## 7) Troubleshooting
```bash
--------------------------------------------------- ---------------------------------------------------------------------------------
Symptom / Error Message                               Cause / Fix
----------------------------------------------------------------------------------------------
ModuleNotFoundError: No module named 'tensorflow'      Not installed in current venv.
                                                       Activate correct environment (which python)
                                                       and run pip install -r requirements.txt.

Metal device not found                                 Running Intel/x86 Terminal on macOS.
                                                       Open native arm64 Terminal and reinstall:
                                                       brew reinstall tensorflow-macos tensorflow-metal

Training extremely slow on M1/M2                       Using new Keras optimizer.
                                                       In model.py, use:
                                                       tf.keras.optimizers.legacy.Adam()

IndentationError: unindent does not match any outer
indentation level                                     Mixed tabs/spaces after code edits.
                                                       Align with 4 spaces or run:
                                                       expand -t 4 src/fruits360/train.py > tmp && mv tmp src/fruits360/train.py

UnboundLocalError: cannot access local variable 'os'   Local import shadows global import.
                                                       Remove local import os, keep only import time.

No best.keras found (but file exists)                  Filesystem async write delay.
                                                       Add fsync() or rerun (already patched).

Memory hits 100% on Ubuntu                             Dataset cache/prefetch fills RAM.
                                                       export FRUITS360_CACHE=0
                                                       export FRUITS360_SHUFFLE_BUFFER=2048
                                                       Monitor with htop.

GPU/CPU threads oversubscribed                         Too many TensorFlow threads.
                                                       Auto tuned: OMP=7, INTRA=7, INTER=2
                                                       Can override manually if needed.

No class_names.json produced                           Older version skipped saving.
                                                       Fixed: now written in train.py to artifacts/class_names.json.

Validation stops early (≈12 epochs)                    EarlyStopping triggered (no improvement).
                                                       Increase patience or lower learning rate.

Shape must be rank 3 but is rank 4                     tf.pad applied after batching.
                                                       Move padding before batching in data.py.

Module fruits360 not found                             Package not installed in editable mode.
                                                       Run pip install -e . from project root.

Logs mixed between CPU/GPU runs                        Both write to same artifact root.
                                                       Use separate folders: artifacts-cpu, artifacts-gpu.
--------------------------------------------------- ---------------------------------------------------------------------------------
```


## 8) Revision History:
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
#2.2.3 Auto scaling for CPU threads and batch size.
#2.2.4 Restore padding in agumentation.
```

## 9) References
```bash
• TensorFlow Metal Plugin Guide
• Keras API Docs
• Fruits 360 Dataset (Kaggle)
```
## 10)Git backup with Tag creation and push to Github.
```bash
bash scripts/snapshot_baseline.sh
git push origin main --tags
git status
```
## 11) Restore or clone from GitHub
```bash
git clone https://github.com/<you>/fruits360-ml-gpu.git
cd fruits360-ml-gpu
git checkout baseline-[revision]
python3 -m venv .venv && source .venv/bin/activate
pip install -r artifacts/requirements.txt
python -m fruits360.infer --image ...
```
