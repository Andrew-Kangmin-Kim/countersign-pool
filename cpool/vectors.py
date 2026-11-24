"""Fixed-length word vectors for the alignment-free metrics (Manhattan,
Euclidean, cosine) of the metric-selection experiment.

Layout: 3 syllables x (onset, vowel, coda) x (4 notebook features + present
bit), syllables left-aligned, absent slots all zero.  The present bit is
needed because ㅂ codes as (0,0,0,0) and would otherwise equal an empty slot.
Same surface form, same features and same silent-ㅇ / cluster-coda handling as
the edit distance (cpool.phonetics).
"""
from __future__ import annotations

import numba as nb
import numpy as np

from .hangul import decompose
from .phonetics import CLUSTER_FINALS, CONS_FEATURES, VOWEL_FEATURES

N_SYL = 3
SLOT = 5                     # 4 features + present bit
DIM = N_SYL * 3 * SLOT       # 45
METRICS = ("manhattan", "euclidean", "cosine")


def encode_one(surface: str) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    if len(surface) > N_SYL:
        raise ValueError(f"{surface!r} has more than {N_SYL} syllables")
    for s, ch in enumerate(surface):
        d = decompose(ch)
        if d is None:
            raise ValueError(f"non-syllable character in surface form: {surface!r}")
        ini, med, fin = d
        slots = [CONS_FEATURES[ini] if ini != "ㅇ" else None,
                 VOWEL_FEATURES[med],
                 CONS_FEATURES[CLUSTER_FINALS.get(fin, fin)] if fin else None]
        for k, f in enumerate(slots):
            if f is not None:
                o = (s * 3 + k) * SLOT
                v[o:o + 4] = f
                v[o + 4] = 1.0
    return v


def encode(surfaces) -> np.ndarray:
    return np.stack([encode_one(s) for s in surfaces])


@nb.njit(parallel=True, cache=True)
def _rows(V, i0, i1, kind):
    n, d = V.shape
    out = np.full((i1 - i0, n), np.inf, np.float32)
    for r in nb.prange(i1 - i0):
        i = i0 + r
        for j in range(i + 1, n):
            acc = 0.0
            if kind == 0:
                for k in range(d):
                    acc += abs(V[i, k] - V[j, k])
            elif kind == 1:
                for k in range(d):
                    t = V[i, k] - V[j, k]
                    acc += t * t
                acc = np.sqrt(acc)
            else:
                dot = 0.0
                na = 0.0
                nb_ = 0.0
                for k in range(d):
                    dot += V[i, k] * V[j, k]
                    na += V[i, k] * V[i, k]
                    nb_ += V[j, k] * V[j, k]
                acc = 1.0 - dot / np.sqrt(na * nb_)
            out[r, j] = acc
    return out


def pair_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "manhattan":
        return float(np.abs(a - b).sum())
    if metric == "euclidean":
        return float(np.sqrt(((a - b) ** 2).sum()))
    if metric == "cosine":
        return float(1.0 - a @ b / np.sqrt((a @ a) * (b @ b)))
    raise ValueError(metric)


def closest_pairs(V: np.ndarray, metric: str, k: int, ok_word: np.ndarray, group: np.ndarray,
                  chunk: int = 512, include_ties: bool = False):
    """The k closest pairs (i<j) with both words allowed (ok_word) and
    group[i] != group[j] (group = surface-form id: identical surface forms are
    excluded).  Ties broken by (d, i, j); include_ties=True also returns every
    pair tied with the k-th.  -> list of (i, j, d)."""
    kind = METRICS.index(metric)
    V = np.ascontiguousarray(V, np.float32)
    best_d = np.empty(0, np.float32)
    best_i = np.empty(0, np.int64)
    best_j = np.empty(0, np.int64)
    for i0 in range(0, len(V), chunk):
        i1 = min(i0 + chunk, len(V))
        D = _rows(V, i0, i1, kind)
        D[~ok_word[i0:i1], :] = np.inf
        D[:, ~ok_word] = np.inf
        D[group[i0:i1, None] == group[None, :]] = np.inf
        flat = D.ravel()
        m = min(k, int(np.isfinite(flat).sum()))
        if m == 0:
            continue
        sel = np.argpartition(flat, m - 1)[:m]
        sel = sel[np.isfinite(flat[sel])]
        # keep everything tied with the k-th value so the final tie-break is exact
        kth = flat[sel].max()
        sel = np.flatnonzero(flat <= kth)
        r, c = np.divmod(sel, D.shape[1])
        best_d = np.concatenate([best_d, flat[sel]])
        best_i = np.concatenate([best_i, r + i0])
        best_j = np.concatenate([best_j, c])
        o = np.lexsort((best_j, best_i, best_d))[: max(k, 1)]
        kth = best_d[o[-1]]
        keep = best_d <= kth
        best_d, best_i, best_j = best_d[keep], best_i[keep], best_j[keep]
    o = np.lexsort((best_j, best_i, best_d))
    o = o[best_d[o] <= best_d[o[k - 1]]] if include_ties and len(o) >= k else o[:k]
    return [(int(best_i[t]), int(best_j[t]), float(best_d[t])) for t in o]


def edge_list(V: np.ndarray, metric: str, tau_max: float, chunk: int = 512, verbose=True):
    """All pairs (i<j) with d <= tau_max under a vector metric -> (i, j, d)."""
    kind = METRICS.index(metric)
    V = np.ascontiguousarray(V, np.float32)
    ei, ej, ed = [], [], []
    for i0 in range(0, len(V), chunk):
        i1 = min(i0 + chunk, len(V))
        D = _rows(V, i0, i1, kind)
        r, c = np.where(D <= tau_max)
        ei.append((r + i0).astype(np.int32))
        ej.append(c.astype(np.int32))
        ed.append(D[r, c])
        if verbose and (i1 == len(V) or (i1 // chunk) % 10 == 0):
            print(f"  distance rows {i1}/{len(V)}", flush=True)
    return np.concatenate(ei), np.concatenate(ej), np.concatenate(ed)
