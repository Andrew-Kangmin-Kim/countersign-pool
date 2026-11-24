"""Lexicon: corpus -> Kiwi NNG lemmas (hard counts) -> 2-3 syllables -> Zipf cutoff.

Three steps.  Hard counts: each eojeol's full count goes
to every NNG in Kiwi's single best analysis (the notebook's method).
Zipf = log10(count per billion corpus tokens).
Cutoff at x = the x-th percentile of Zipf over NIKL 2003 level-C nouns present
in the corpus (absent anchors are dropped).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HANGUL_23 = None


def load_corpus(path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 2 and p[1].isdigit():
                rows.append((p[0], int(p[1])))
    return pd.DataFrame(rows, columns=["eojeol", "count"])


def lemmatize(corpus: pd.DataFrame) -> pd.DataFrame:
    from kiwipiepy import Kiwi
    kiwi = Kiwi(num_workers=-1)
    freq: dict[str, int] = {}
    counts = corpus["count"].to_numpy()
    for k, toks in enumerate(kiwi.tokenize(corpus["eojeol"].tolist())):
        for lem in {t.form for t in toks if t.tag == "NNG"}:
            freq[lem] = freq.get(lem, 0) + int(counts[k])
    return pd.DataFrame(sorted(freq.items()), columns=["lemma", "count"])


def anchors(path, level="C", pos="명") -> list[str]:
    raw = Path(path).read_bytes().decode("cp949")
    out = set()
    for line in raw.splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 5 and c[2].strip() == pos and c[4].strip() == level:
            out.add(re.sub(r"\d+$", "", c[1].strip()))
    return sorted(out)


def zipf(count, total):
    return np.log10(np.asarray(count, float) / total * 1e9)
