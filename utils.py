import random


def subject_split(subjects, seed=42, train=0.70, val=0.15):
    """Split by SUBJECT (not by image) so no person appears in two splits.
    Returns {subject: 'train'|'val'|'test'}."""
    subjects = sorted(set(subjects))
    random.Random(seed).shuffle(subjects)
    n = len(subjects)
    n_tr, n_va = round(n * train), round(n * val)
    out = {}
    for i, s in enumerate(subjects):
        out[s] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return out
