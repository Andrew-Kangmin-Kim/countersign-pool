"""Articulatory features (the notebook's 4-dim coding) and the weighted
Manhattan edit distance (substitution = weighted L1 within type, CROSS across
consonant/vowel, GAP per insertion/deletion).  Words are encoded from their
g2pk2 surface form; silent initial ㅇ is omitted.  Distance >= |len diff| * GAP,
which licenses length blocking in the edge list.
"""
from __future__ import annotations

import math

import numba as nb
import numpy as np

from .hangul import decompose

NFEAT = 4
CONS_FEATURES = {  # (place, manner, laryngeal, sonorant)
    "ㄱ": (3, 0, 0, 0), "ㄲ": (3, 0, 1, 0), "ㅋ": (3, 0, 2, 0),
    "ㄷ": (1, 0, 0, 0), "ㄸ": (1, 0, 1, 0), "ㅌ": (1, 0, 2, 0),
    "ㅂ": (0, 0, 0, 0), "ㅃ": (0, 0, 1, 0), "ㅍ": (0, 0, 2, 0),
    "ㅈ": (2, 3, 0, 0), "ㅉ": (2, 3, 1, 0), "ㅊ": (2, 3, 2, 0),
    "ㅅ": (1, 2, 0, 0), "ㅆ": (1, 2, 1, 0), "ㅎ": (4, 2, 0, 0),
    "ㅁ": (0, 1, 0, 1), "ㄴ": (1, 1, 0, 1), "ㅇ": (3, 1, 0, 1), "ㄹ": (1, 4, 0, 1),
}
VOWEL_FEATURES = {  # (height, backness, rounding, glide)
    "ㅏ": (2, 2, 0, 0), "ㅐ": (2, 0, 0, 0), "ㅑ": (2, 2, 0, 1), "ㅒ": (2, 0, 0, 1),
    "ㅓ": (1, 2, 0, 0), "ㅔ": (1, 0, 0, 0), "ㅕ": (1, 2, 0, 1), "ㅖ": (1, 0, 0, 1),
    "ㅗ": (1, 2, 1, 0), "ㅘ": (2, 2, 0, 2), "ㅙ": (2, 0, 0, 2), "ㅚ": (1, 0, 1, 0),
    "ㅛ": (1, 2, 1, 1), "ㅜ": (0, 2, 1, 0), "ㅝ": (1, 2, 0, 2), "ㅞ": (1, 0, 0, 2),
    "ㅟ": (0, 0, 1, 2), "ㅠ": (0, 2, 1, 1), "ㅡ": (0, 1, 0, 0), "ㅢ": (0, 1, 0, 1),
    "ㅣ": (0, 0, 0, 0),
}
CLUSTER_FINALS = {"ㄳ": "ㄱ", "ㄵ": "ㄴ", "ㄶ": "ㄴ", "ㄺ": "ㄱ", "ㄻ": "ㅁ", "ㄼ": "ㄹ",
                  "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㄿ": "ㅂ", "ㅀ": "ㄹ", "ㅄ": "ㅂ"}
FEATURE_NAMES = ["c_place", "c_manner", "c_laryngeal", "c_sonorant",
                 "v_height", "v_backness", "v_round", "v_glide", "n_gap", "n_cross"]


def phonemes(surface: str) -> list[tuple[int, tuple]]:
    """surface form -> [(type, features)]; silent initial ㅇ omitted."""
    seq = []
    for ch in surface:
        d = decompose(ch)
        if d is None:
            raise ValueError(f"non-syllable character in surface form: {surface!r}")
        ini, med, fin = d
        if ini != "ㅇ":
            seq.append((0, CONS_FEATURES[ini]))
        seq.append((1, VOWEL_FEATURES[med]))
        if fin:
            seq.append((0, CONS_FEATURES[CLUSTER_FINALS.get(fin, fin)]))
    return seq


class Encoded:
    """Fixed-width arrays so numba kernels never touch Python objects."""

    def __init__(self, surfaces: list[str], maxlen: int = 9):
        n = len(surfaces)
        self.TYPE = np.full((n, maxlen), -1, np.int8)
        self.FEAT = np.zeros((n, maxlen, NFEAT), np.float32)
        self.LEN = np.zeros(n, np.int32)
        for k, s in enumerate(surfaces):
            seq = phonemes(s)
            if len(seq) > maxlen:
                raise ValueError(f"{s!r} has {len(seq)} phonemes > maxlen={maxlen}")
            self.LEN[k] = len(seq)
            for p, (t, f) in enumerate(seq):
                self.TYPE[k, p] = t
                self.FEAT[k, p] = f

    def subset(self, idx):
        e = Encoded.__new__(Encoded)
        e.TYPE, e.FEAT, e.LEN = self.TYPE[idx], self.FEAT[idx], self.LEN[idx]
        return e

    def __len__(self):
        return len(self.LEN)


class Params:
    """Distance costs: 4 consonant + 4 vowel feature weights, gap, cross."""

    def __init__(self, cons_w, vowel_w, gap, cross, source=""):
        self.cons_w = np.asarray(cons_w, np.float32)
        self.vowel_w = np.asarray(vowel_w, np.float32)
        self.gap = float(gap)
        self.cross = float(cross)
        self.source = source

    def to_dict(self):
        return {"cons_w": self.cons_w.tolist(), "vowel_w": self.vowel_w.tolist(),
                "gap": self.gap, "cross": self.cross, "source": self.source}

    @classmethod
    def from_dict(cls, d):
        return cls(d["cons_w"], d["vowel_w"], d["gap"], d["cross"], d.get("source", ""))

    def args(self):
        return (self.cons_w, self.vowel_w, np.float32(self.gap), np.float32(self.cross))


# ------------------------------------------------------------------ kernels
@nb.njit(cache=True)
def _sub(tA, fA, tB, fB, cw, vw, cross):
    if tA != tB:
        return cross
    w = cw if tA == 0 else vw
    acc = 0.0
    for k in range(w.shape[0]):
        acc += w[k] * abs(fA[k] - fB[k])
    return acc


@nb.njit(cache=True)
def pair_dist(tA, fA, lA, tB, fB, lB, cw, vw, gap, cross):
    D = np.empty((lA + 1, lB + 1), np.float32)
    for i in range(lA + 1):
        D[i, 0] = i * gap
    for j in range(lB + 1):
        D[0, j] = j * gap
    for i in range(1, lA + 1):
        for j in range(1, lB + 1):
            s = D[i - 1, j - 1] + _sub(tA[i - 1], fA[i - 1], tB[j - 1], fB[j - 1], cw, vw, cross)
            d1 = D[i - 1, j] + gap
            d2 = D[i, j - 1] + gap
            if d1 < s:
                s = d1
            if d2 < s:
                s = d2
            D[i, j] = s
    return D[lA, lB]


@nb.njit(parallel=True, cache=True)
def _chunk(TYPE, FEAT, LEN, i0, i1, cw, vw, gap, cross, len_diff_max, n_limit):
    out = np.full((i1 - i0, n_limit), np.inf, np.float32)
    for r in nb.prange(i1 - i0):
        i = i0 + r
        for j in range(i + 1, n_limit):
            if abs(LEN[i] - LEN[j]) <= len_diff_max:
                out[r, j] = pair_dist(TYPE[i], FEAT[i], LEN[i], TYPE[j], FEAT[j], LEN[j],
                                      cw, vw, gap, cross)
    return out


@nb.njit(parallel=True, cache=True)
def _cross(TA, FA, LA, TB, FB, LB, cw, vw, gap, cross):
    out = np.empty((TA.shape[0], TB.shape[0]), np.float32)
    for i in nb.prange(TA.shape[0]):
        for j in range(TB.shape[0]):
            out[i, j] = pair_dist(TA[i], FA[i], LA[i], TB[j], FB[j], LB[j], cw, vw, gap, cross)
    return out


def cross_distances(A: Encoded, B: Encoded, P: Params) -> np.ndarray:
    return _cross(A.TYPE, A.FEAT, A.LEN, B.TYPE, B.FEAT, B.LEN, *P.args())


def edge_list(E: Encoded, P: Params, tau_max: float, chunk: int = 512, verbose=True):
    """All pairs (i<j) with d < tau_max.  Pairs with |len diff| * gap >= tau_max
    are certified at or above tau_max without alignment."""
    n = len(E)
    len_diff_max = int(math.ceil(tau_max / P.gap) - 1) if P.gap > 0 else 99
    ei, ej, ed = [], [], []
    aligned = 0
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        D = _chunk(E.TYPE, E.FEAT, E.LEN, i0, i1, *P.args()[:2], np.float32(P.gap),
                   np.float32(P.cross), len_diff_max, n)
        aligned += int(np.isfinite(D).sum())
        r, c = np.where(D < tau_max)
        ei.append((r + i0).astype(np.int32))
        ej.append(c.astype(np.int32))
        ed.append(D[r, c])
        if verbose:
            print(f"\r  distance rows {i1}/{n}", end="", flush=True)
    if verbose:
        print()
    return (np.concatenate(ei), np.concatenate(ej), np.concatenate(ed),
            {"n": n, "pairs": n * (n - 1) // 2, "aligned": aligned,
             "len_diff_max": len_diff_max, "tau_max": tau_max})


# --------------------------------------------------------- alignment features
def feature_diff(sa: str, sb: str, P: Params) -> np.ndarray:
    """Backtrace the optimal alignment UNDER P and tally the 10 cost components
    (8 feature differences, #gaps, #cross substitutions).  Then
    distance(sa, sb; P) == diff @ [cons_w, vowel_w, gap, cross]."""
    A, B = phonemes(sa), phonemes(sb)
    m, n = len(A), len(B)
    cw, vw = P.cons_w.astype(float), P.vowel_w.astype(float)
    D = np.zeros((m + 1, n + 1))
    bp = np.zeros((m + 1, n + 1), np.int8)
    for i in range(1, m + 1):
        D[i, 0] = i * P.gap
        bp[i, 0] = 1
    for j in range(1, n + 1):
        D[0, j] = j * P.gap
        bp[0, j] = 2
    for i in range(1, m + 1):
        ta, fa = A[i - 1]
        for j in range(1, n + 1):
            tb, fb = B[j - 1]
            if ta != tb:
                sc = P.cross
            else:
                w = cw if ta == 0 else vw
                sc = float(np.dot(w, np.abs(np.subtract(fa, fb))))
            sub, dele, ins = D[i - 1, j - 1] + sc, D[i - 1, j] + P.gap, D[i, j - 1] + P.gap
            best = min(sub, dele, ins)
            D[i, j] = best
            bp[i, j] = 0 if best == sub else (1 if best == dele else 2)
    x = np.zeros(10)
    i, j = m, n
    while i > 0 or j > 0:
        mv = bp[i, j]
        if mv == 0:
            (ta, fa), (tb, fb) = A[i - 1], B[j - 1]
            if ta != tb:
                x[9] += 1
            elif ta == 0:
                x[0:4] += np.abs(np.subtract(fa, fb))
            else:
                x[4:8] += np.abs(np.subtract(fa, fb))
            i -= 1
            j -= 1
        else:
            x[8] += 1
            if mv == 1:
                i -= 1
            else:
                j -= 1
    return x


def param_vector(P: Params) -> np.ndarray:
    return np.concatenate([P.cons_w, P.vowel_w, [P.gap, P.cross]]).astype(float)
