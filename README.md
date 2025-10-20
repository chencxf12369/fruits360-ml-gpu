# Fruits-360 (CPU-only, TensorFlow 2.16, Python 3.10)

**CPU target:** Intel i9-10980XE (18C/36T)  
**Python:** /usr/local/bin/python3.10

## 1) Setup
```bash
bash scripts/setup_env.sh
bash scripts/download_data.sh

2) Train
source .venv/bin/activate
export FRUITS360_BACKBONE=mobilenetv2
export FRUITS360_FREEZE_BASE=1   # freeze backbone (fast on CPU)
#export FRUITS360_FREEZE_BASE=0   # unfreeze for fine-tuning (slower but +accuracy)
#For staged training, first run with FREEZE_BASE=1 for quick convergence, 
#then re-run with FREEZE_BASE=0 and a lower LR (already handled in model.py: 1e-3 when frozen → 1e-4 when unfrozen).
##full cleanup if necessary
##rm -rf src/fruits360.egg-info
##pip uninstall fruits360 -y

cd /root/ml-gpu
pip install -e .

python -m fruits360.train


Artifacts land in ./artifacts/:

fruits360_best.keras

fruits360_savedmodel/

train_log.csv

3) Evaluate on official Test split
python -m fruits360.eval




4) Inference
python -m fruits360.infer --image "$HOME/data/Fruit-Images-Dataset/Test/Apple Golden 1/0_100.jpg"

Notes

CPU-only enforced by env and tf.config.set_visible_devices([], "GPU").

Threading defaults: OMP_NUM_THREADS=18, TF_NUM_INTRAOP_THREADS=18, TF_NUM_INTEROP_THREADS=2.
Tune per workload; for heavier convs, INTEROP=1–2 and INTRA=cores are good starting points.

Reproduce env: make freeze then reinstall with pip install -r requirements.lock.txt

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

