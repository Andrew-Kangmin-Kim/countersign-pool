"""Calibration — the notebook's procedure (Cells 7-9), with its bugs fixed.

Notebook procedure, kept:
  * sample: 800 random words scanned with uniform weights; members of the 30
    closest pairs + random words up to 160
  * self-recognition gate on clean audio; trials at SNR 0/-5/-10/-15, 8 each
  * decision rule (raw vs prior-corrected): the one whose pair confusion the
    uniform distance predicts better (more negative Spearman)
  * pair rate = confusions / (times u spoken + times v spoken)
  * binomial GLM (freq_weights = trials) of pair rate on the 10 alignment
    features, fit on the nearest 10% of pairs (all pairs if that tail is too
    sparse: < 40 pairs or < 10 confusions)
  * GAP=3 and CROSS=6 stay fixed; weights floored at 0.1; three candidates
    (uniform / raw calibrated / mean-normalized calibrated) and the one with
    the most negative Spearman(distance, pair rate) is adopted

Bugs fixed:
  1. SIGN — notebook used |coef|.  More feature difference must mean LESS
     confusion (negative coef); a wrong-sign coefficient became a large
     weight.  Fixed: weight = max(-coef, floor).
  2. ALIGNMENT — notebook built the design matrix from the uniform-weight
     alignment, then used the weights with a different (weighted) alignment.
     Fixed: re-align under the fitted weights and refit (align_iters times).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from .phonetics import FEATURE_NAMES, Params, feature_diff


def pair_table(trials: pd.DataFrame, words, col) -> pd.DataFrame:
    spoken = trials["true_word"].value_counts().to_dict()
    mis = trials[trials["true_word"] != trials[col]]
    conf = {}
    for u, v in zip(mis["true_word"], mis[col]):
        k = tuple(sorted((u, v)))
        conf[k] = conf.get(k, 0) + 1
    rows = []
    for u, v in itertools.combinations(sorted(words), 2):
        n = spoken.get(u, 0) + spoken.get(v, 0)
        c = conf.get((u, v), 0)
        rows.append((u, v, c, n, c / n if n else 0.0))
    return pd.DataFrame(rows, columns=["w1", "w2", "confusions", "trials", "rate"])


def distances(pairs, surf, P: Params) -> np.ndarray:
    th = np.concatenate([P.cons_w, P.vowel_w, [P.gap, P.cross]]).astype(float)
    return np.array([feature_diff(surf[a], surf[b], P) @ th for a, b in zip(pairs.w1, pairs.w2)])


def _weights_from(coef: pd.Series, floor: float):
    cw, vw = np.ones(4), np.ones(4)
    slot = {n: (cw if i < 4 else vw, i % 4) for i, n in enumerate(FEATURE_NAMES[:8])}
    signs = {}
    for name, val in coef.items():
        if name in slot:
            arr, i = slot[name]
            arr[i] = max(-val, floor)          # FIX 1: sign-aware (notebook: abs(val))
            signs[name] = "ok" if val < 0 else "wrong_sign_floored"
    return cw, vw, signs


def calibrate(pairs: pd.DataFrame, surf: dict, cfg, section: str = "calibration") -> dict:
    c, ph = cfg[section], cfg["phonetics"]
    gap, cross = ph["gap"], ph["cross"]
    uniform = Params(np.ones(4), np.ones(4), gap, cross, source="uniform")
    d_uni = distances(pairs, surf, uniform)
    rho_uni = spearmanr(d_uni, pairs.rate).statistic

    thr = np.quantile(d_uni, c["fit_tail"])
    mask = d_uni <= thr
    if mask.sum() >= 40 and pairs.confusions.values[mask].sum() >= 10:
        idx, scope = np.flatnonzero(mask), f"nearest {c['fit_tail']:.0%} (n={mask.sum()})"
    else:
        idx, scope = np.arange(len(pairs)), f"all pairs (tail too sparse, n={len(pairs)})"

    P = uniform
    history = []
    for it in range(c["align_iters"]):           # FIX 2: alignment under current weights
        X = pd.DataFrame([feature_diff(surf[a], surf[b], P)
                          for a, b in zip(pairs.w1.values[idx], pairs.w2.values[idx])],
                         columns=FEATURE_NAMES)
        nz = X.columns[X.abs().sum() > 0].tolist()
        n = pairs.trials.clip(lower=1).values[idx]
        glm = sm.GLM(pairs.confusions.values[idx] / n, sm.add_constant(X[nz]),
                     family=sm.families.Binomial(), freq_weights=n).fit()
        coef = glm.params.drop("const", errors="ignore")
        cw, vw, signs = _weights_from(coef, c["weight_floor"])
        raw = Params(cw, vw, gap, cross, source="calibrated_raw")
        m = np.concatenate([cw, vw]).mean()
        norm = Params(cw / m, vw / m, gap, cross, source="calibrated_norm")
        history.append({"iter": it, "coef": coef.to_dict(), "signs": signs,
                        "weights_norm": np.concatenate([norm.cons_w, norm.vowel_w]).tolist()})
        if it > 0 and np.allclose(norm.cons_w, P.cons_w, atol=1e-3) and \
                np.allclose(norm.vowel_w, P.vowel_w, atol=1e-3):
            break
        P = norm

    cands = {"uniform": uniform, "calibrated_raw": raw, "calibrated_norm": norm}
    rhos = {k: float(spearmanr(distances(pairs, surf, v), pairs.rate).statistic)
            for k, v in cands.items()}
    rhos["uniform"] = float(rho_uni)
    best = min(rhos, key=rhos.get)                # notebook adoption rule
    return {"params": cands[best], "adopted": best, "spearman": rhos, "fit_scope": scope,
            "glm_summary": glm.summary().as_text(), "history": history,
            "candidates": {k: v.to_dict() for k, v in cands.items()}}
