"""Step 1 — lexicon.  Outputs runs/lexicon/:
  lemmas.csv        every NNG lemma: count, zipf
  universe.csv      the loosest set (x=5%), frequency-sorted, with g2pk2 surface;
                    every larger x is a prefix (n_x in summary.json)
  summary.json      corpus size, anchors, cutoffs, N per x, attrition
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from cpool import config as C
from cpool import g2p, lexicon


def main():
    cfg = C.load()
    out = C.stage(cfg, "lexicon")
    for k in ("corpus", "anchors"):
        h = C.sha256(C.path(cfg["data"][k]))
        assert h == cfg["data"][k + "_sha256"], f"{k} hash mismatch {h}"

    corpus = lexicon.load_corpus(C.path(cfg["data"]["corpus"]))
    total = int(corpus["count"].sum())
    t0 = time.time()
    lem = lexicon.lemmatize(corpus)
    lem["zipf"] = lexicon.zipf(lem["count"], total)
    lem.to_csv(out / "lemmas.csv", index=False, encoding="utf-8-sig")
    print(f"corpus {len(corpus):,} entries / {total:,} tokens -> {len(lem):,} NNG lemmas "
          f"({time.time()-t0:.0f}s)", flush=True)

    anc = lexicon.anchors(C.path(cfg["data"]["anchors"]))
    zmap = dict(zip(lem["lemma"], lem["zipf"]))
    present = [a for a in anc if a in zmap]
    az = np.array([zmap[a] for a in present])
    qs = sorted(float(q) for q in cfg["lexicon"]["quantiles"])
    cut = {q: float(np.percentile(az, q)) for q in qs}
    print(f"anchors: {len(present)}/{len(anc)} level-C nouns present; cutoffs "
          + ", ".join(f"x={q:g}%: Zipf {c:.3f} (= {10**c * total / 1e9:.1f} occurrences)"
                      for q, c in cut.items()), flush=True)

    lo, hi = cfg["lexicon"]["syllables"]
    syl = lem[lem["lemma"].str.fullmatch(r"[가-힣]{%d,%d}" % (lo, hi))]
    uni = syl[syl["zipf"] >= cut[qs[0]]].sort_values(["count", "lemma"], ascending=[False, True])
    uni = uni.reset_index(drop=True)
    sf = g2p.surfaces_parallel(uni["lemma"].tolist())
    uni["surface"] = uni["lemma"].map(sf)
    uni.to_csv(out / "universe.csv", index=False, encoding="utf-8-sig")
    n_x = {str(q): int((uni["zipf"] >= c).sum()) for q, c in cut.items()}
    # prefix property: higher cutoff => strictly a frequency prefix
    for q, c in cut.items():
        assert (uni["zipf"].iloc[: n_x[str(q)]] >= c).all()

    summ = {"corpus_entries": len(corpus), "corpus_tokens": total, "nng_lemmas": len(lem),
            "after_syllables": len(syl),
            "anchors_total": len(anc), "anchors_present": len(present),
            "cutoff_zipf": {str(q): c for q, c in cut.items()},
            "cutoff_occurrences": {str(q): 10 ** c * total / 1e9 for q, c in cut.items()},
            "N_x": n_x, "headline": cfg["lexicon"]["headline"],
            "surface_differs_from_spelling": int((uni["surface"] != uni["lemma"]).sum())}
    C.save(summ, out / "summary.json")
    C.manifest(cfg, "lexicon", summ)
    print(summ)
    print("sample (x=5%):", " ".join(uni["lemma"].sample(40, random_state=cfg["seed"])))


if __name__ == "__main__":
    main()
