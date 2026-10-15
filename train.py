"""Step 2: train a classifier.
  python train.py --manifest data/mrl_manifest.csv   --task eye  --arch improved
  python train.py --manifest data/yawn_manifest.csv  --task yawn --arch improved
Run each task with --arch simple / improved / mobilenet to fill the comparison table.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import config, models
from data import make_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task", required=True, choices=["eye", "yawn"])
    ap.add_argument("--arch", default="improved", choices=["simple", "improved", "mobilenet"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    keras.utils.set_random_seed(config.SEED)

    df = pd.read_csv(a.manifest)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if tr.empty or va.empty:
        raise SystemExit("Manifest must contain non-empty train and val splits")
    n, n1 = len(tr), int(tr.label.sum())
    if n1 == 0 or n1 == n:
        raise SystemExit("Training split must contain both label classes (0 and 1)")
    cw = {0: n / (2 * (n - n1)), 1: n / (2 * n1)}
    print("train/val:", len(tr), len(va), "class weights:", cw)

    name = f"{a.task}_{a.arch}"
    Path("outputs").mkdir(parents=True, exist_ok=True)
    model = models.build(a.arch)
    model.compile(keras.optimizers.Adam(a.lr), "binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall"),
                           keras.metrics.AUC(name="auc")])
    cbs = [keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2),
           keras.callbacks.ModelCheckpoint(f"outputs/{name}.keras", "val_loss", save_best_only=True)]
    h = model.fit(make_ds(tr, True, a.batch), validation_data=make_ds(va, False, a.batch),
                  epochs=a.epochs, class_weight=cw, callbacks=cbs, shuffle=False)
    model.save(f"outputs/{name}.keras")
    json.dump(h.history, open(f"outputs/{name}_history.json", "w"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for k in ("loss", "val_loss"):
        ax[0].plot(h.history[k], label=k)
    for k in ("accuracy", "val_accuracy"):
        ax[1].plot(h.history[k], label=k)
    for x, t in zip(ax, ("Loss", "Accuracy")):
        x.set_title(f"{name} - {t}"); x.set_xlabel("epoch"); x.legend()
    fig.tight_layout(); fig.savefig(f"outputs/{name}_curves.png", dpi=150)
    print("saved outputs/%s.keras" % name)


if __name__ == "__main__":
    main()
