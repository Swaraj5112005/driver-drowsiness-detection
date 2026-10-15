"""Step 1b: extract mouth crops + weak yawn labels from YawDD videos.

YawDD facts used here:
 * Dash set: several videos per driver, named like 1-FemaleNoGlasses-Normal / -Talking / -Yawning
 * Mirror set: ONE video per driver containing normal + talking + yawning (no condition suffix)
Labelling rule (weak labels - spot check them!):
 * Normal / Talking videos -> every frame is label 0 (talking = useful hard negatives)
 * Yawning / mixed videos  -> label 1 if mouth-aspect-ratio (MAR) stays > HIGH for 3 sampled frames,
                              label 0 if MAR stays < LOW, otherwise frame skipped (ambiguous)
"""
import argparse
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from landmarks import FaceLandmarker, mouth_crop, mar
from utils import subject_split
import config

HIGH, LOW = 0.50, 0.35


def parse(p: Path):
    toks = p.stem.split("-")
    set_name = "dash" if "dash" in str(p).lower() else "mirror"
    gender = "F" if "female" in p.stem.lower() else "M"
    cond = toks[-1].lower() if len(toks) >= 3 else "mixed"
    return f"{set_name}_{gender}_{toks[0]}", cond


def process(path, stride):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise OSError(f"Could not open video: {path}")
    face = None
    items, i = [], 0
    try:
        face = FaceLandmarker(static=False)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                pts = face(frame)
                if pts is not None:
                    crop = mouth_crop(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), pts)
                    if crop is not None:
                        items.append((i, float(mar(pts)), crop))
            i += 1
    finally:
        cap.release()
        if face is not None:
            face.mesh.close()
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with YawDD .avi files (Dash and/or Mirror)")
    ap.add_argument("--out", default="data/yawn")
    ap.add_argument("--manifest", default="data/yawn_manifest.csv")
    ap.add_argument("--stride", type=int, default=5, help="keep every Nth frame")
    ap.add_argument("--max-neg", type=int, default=150, help="max negatives kept per video")
    a = ap.parse_args()
    if a.stride < 1:
        ap.error("--stride must be at least 1")
    if a.max_neg < 0:
        ap.error("--max-neg cannot be negative")

    videos = sorted(Path(a.root).rglob("*.avi"))
    if not videos:
        raise SystemExit("No .avi files found - check --root")
    split_map = subject_split([parse(v)[0] for v in videos], config.SEED)
    rnd = random.Random(config.SEED)
    out, rows = Path(a.out), []
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)

    for n, v in enumerate(videos, 1):
        subj, cond = parse(v)
        split = split_map[subj]
        items = process(v, a.stride)
        if not items:
            print(f"[{n}/{len(videos)}] {v.name}: no faces"); continue
        m = np.array([x[1] for x in items])
        labels = []
        for k in range(len(items)):
            win = m[max(0, k - 1):k + 2]
            if cond in ("normal", "talking"):
                labels.append(0)
            elif win.min() > HIGH:
                labels.append(1)
            elif win.max() < LOW:
                labels.append(0)
            else:
                labels.append(None)
        pos = [k for k, l in enumerate(labels) if l == 1]
        neg = [k for k, l in enumerate(labels) if l == 0]
        rnd.shuffle(neg)
        for k in pos + neg[:a.max_neg]:
            d = out / split / str(labels[k]); d.mkdir(parents=True, exist_ok=True)
            fp = d / f"{v.stem}_{items[k][0]:06d}.jpg"
            cv2.imwrite(str(fp), items[k][2])
            rows.append(dict(path=str(fp.resolve()), label=labels[k], subject=subj,
                             split=split, video=v.name, mar=items[k][1]))
        print(f"[{n}/{len(videos)}] {v.name}: {len(pos)} yawn, {min(len(neg), a.max_neg)} non-yawn")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No face crops were extracted; check the videos and face visibility")
    df.to_csv(a.manifest, index=False)
    print(df.groupby("split").agg(images=("label", "size"), yawn_frac=("label", "mean")))


if __name__ == "__main__":
    main()
