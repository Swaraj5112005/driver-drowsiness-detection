"""Step 3: test-set evaluation + robustness slices + latency.
  python evaluate.py --manifest data/mrl_manifest.csv --model outputs/eye_improved.keras
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)
from tensorflow import keras
from data import make_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()

    name = os.path.splitext(os.path.basename(a.model))[0]
    te = pd.read_csv(a.manifest).query("split == 'test'").reset_index(drop=True)
    if te.empty:
        raise SystemExit("Manifest must contain a non-empty test split")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(a.model)
    prob = model.predict(make_ds(te, False), verbose=0).ravel()
    pred = (prob >= a.threshold).astype(int)
    y = te.label.values

    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred)
    res = dict(model=name, n_test=len(te), accuracy=accuracy_score(y, pred), precision=p,
               recall=r, f1=f1, auc=roc_auc_score(y, prob) if len(set(y)) > 1 else None)

    # robustness slices (MRL has glasses / lighting / reflections / gender metadata)
    te["correct"] = (pred == y)
    for col in ("glasses", "lighting", "reflections", "gender"):
        if col in te:
            res[f"acc_by_{col}"] = te.groupby(col).correct.mean().round(4).to_dict()

    # latency: single image, batch size 1
    x = np.zeros((1, 64, 64, 1), "float32")
    for _ in range(20):
        model(x, training=False)
    t0 = time.perf_counter()
    for _ in range(200):
        model(x, training=False)
    ms = (time.perf_counter() - t0) / 200 * 1000
    res.update(latency_ms=round(ms, 2), fps=round(1000 / ms, 1))

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(name)
    fig.tight_layout(); fig.savefig(f"outputs/{name}_cm.png", dpi=150)

    # save the worst mistakes' paths for error analysis
    te.assign(prob=prob)[~te.correct].assign(conf=lambda d: abs(d.prob - 0.5)) \
        .sort_values("conf", ascending=False).head(50)[["path", "label", "prob"]] \
        .to_csv(f"outputs/{name}_errors.csv", index=False)

    json.dump(res, open(f"outputs/{name}_report.json", "w"), indent=2, default=float)
    row = {k: v for k, v in res.items() if not k.startswith("acc_by")}
    pd.DataFrame([row]).to_csv("outputs/summary.csv", mode="a", index=False,
                               header=not os.path.exists("outputs/summary.csv"))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
