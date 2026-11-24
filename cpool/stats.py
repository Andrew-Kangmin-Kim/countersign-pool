"""Small statistics helpers shared by the experiments."""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def separation(top, ctrl, B=2000, seed=0) -> dict:
    """ratio of means, Mann-Whitney (top > ctrl), Cliff's delta; bootstrap CIs
    resampling pairs within each group."""
    top, ctrl = np.asarray(top, float), np.asarray(ctrl, float)
    rng = np.random.RandomState(seed)
    ratio = top.mean() / max(ctrl.mean(), 1e-12)
    try:
        p = float(mannwhitneyu(top, ctrl, alternative="greater").pvalue)
    except ValueError:
        p = float("nan")
    delta = cliffs_delta(top, ctrl)
    rb, db = [], []
    for _ in range(B):
        t = top[rng.randint(len(top), size=len(top))]
        c = ctrl[rng.randint(len(ctrl), size=len(ctrl))]
        rb.append(t.mean() / max(c.mean(), 1e-12) if c.mean() > 0 else np.nan)
        db.append(cliffs_delta(t, c))
    rb = np.array(rb)
    return {"n_top": len(top), "n_ctrl": len(ctrl),
            "mean_top": float(top.mean()), "mean_ctrl": float(ctrl.mean()),
            "median_top": float(np.median(top)), "median_ctrl": float(np.median(ctrl)),
            "ratio": float(ratio),
            "ratio_ci": [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))],
            "ratio_boot_undefined_share": float(np.isnan(rb).mean()),
            "mw_p": p, "cliffs_delta": delta,
            "delta_ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))]}


def boot_mean_ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.RandomState(seed)
    m = [x[rng.randint(len(x), size=len(x))].mean() for _ in range(B)]
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
