"""Step 1a: build a subject-aware manifest for the MRL Eye dataset.
MRL filename: s0001_00123_0_0_0_0_0_01.png
  subject_imageID_gender_glasses_eyeState_reflections_lighting_sensor
  eyeState: 0=closed, 1=open   -> we use label 1 = CLOSED (the drowsy class)
"""
import argparse
from pathlib import Path
import pandas as pd
from utils import subject_split
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing MRL .png files")
    ap.add_argument("--out", default="data/mrl_manifest.csv")
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in Path(a.root).rglob("*.png"):
        f = p.stem.split("_")
        if len(f) < 8:
            continue
        rows.append(dict(path=str(p.resolve()), subject=f[0], gender=int(f[2]),
                         glasses=int(f[3]), label=1 - int(f[4]),
                         reflections=int(f[5]), lighting=int(f[6]), sensor=f[7]))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No MRL images found - check --root")
    df["split"] = df.subject.map(subject_split(df.subject.unique(), config.SEED))
    df.to_csv(a.out, index=False)
    print(f"{len(df)} images, {df.subject.nunique()} subjects")
    print(df.groupby("split").agg(images=("label", "size"), subjects=("subject", "nunique"),
                                  closed_frac=("label", "mean")))


if __name__ == "__main__":
    main()
