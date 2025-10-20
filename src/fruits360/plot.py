from __future__ import annotations
import json, sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from . import config

def main() -> int:
    csv_path = config.HISTORY_CSV
    if not csv_path.exists():
        print(f"[plot] Missing history CSV: {csv_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(csv_path)
    epoch = df.index + 1
    # accuracy
    plt.figure(figsize=(7,5))
    for c in [c for c in df.columns if c.lower() in {"accuracy","acc","val_accuracy","val_acc"}]:
        plt.plot(epoch, df[c], label=c)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend(); plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout(); plt.savefig(config.PLOT_ACC_PATH, dpi=150); plt.close()
    # loss
    plt.figure(figsize=(7,5))
    for c in [c for c in df.columns if c.lower() in {"loss","val_loss"}]:
        plt.plot(epoch, df[c], label=c)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend(); plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout(); plt.savefig(config.PLOT_LOSS_PATH, dpi=150); plt.close()
    print(f"[plot] Wrote:\n - {config.PLOT_ACC_PATH}\n - {config.PLOT_LOSS_PATH}")
    return 0

if __name__ == "__main__":  # allows `python src/fruits360/plot.py` too
    raise SystemExit(main())
