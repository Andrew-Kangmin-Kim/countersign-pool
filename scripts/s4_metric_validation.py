"""Step 4 — metric validation: closest 100 vs random 100 pairs, on the original
channel and on the held-out channel (notebook Cell 8C, bugs fixed).

  CLOSEST  the metric-closest pairs in the x=5% universe (notebook drew them
           from a 1,200-word subsample but the controls from everything —
           fixed: both from the same set)
  RANDOM   random pairs matched to CLOSEST's syllable-length profile (notebook)
Calibration and pilot words are excluded.  A word that fails the clean
gate (vs 50 random distractors, never its partner) on EITHER channel removes
its pair; the next ranked/random pair replaces it (notebook silently scored
such pairs as 0, and tested words against their own partners — both fixed).
Pair rate = (#a heard as b + #b heard as a) / (#a spoken + #b spoken), decision
rule = the one chosen in calibration.  Runs both channels (use a GPU).
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cpool import config as C
from cpool.channel import build_channel
from cpool.stats import separation


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recover-from-cache", action="store_true")
    args = ap.parse_args()
    cfg = C.load()
    mv = cfg["metric_validation"]
    out = C.stage(cfg, "metric_validation")
    uni = pd.read_csv(C.stage(cfg, "lexicon") / "universe.csv", encoding="utf-8-sig")
    words = uni["lemma"].tolist()
    cal = C.load_json(C.stage(cfg, "calibration") / "result.json")
    rule = cal["decision_rule"]
    excl = set(C.load_json(C.stage(cfg, "calibration") / "words.json")["sample"])
    pw = C.stage(cfg, "pilot_heldout") / "words.json"
    if pw.exists():
        excl |= set(C.load_json(pw)["pilot_words"])
    z = np.load(C.stage(cfg, "distances") / "edges.npz")
    EI, EJ, ED = z["i"], z["j"], z["d"]

    keep = np.array([words[i] not in excl and words[j] not in excl for i, j in zip(EI, EJ)])
    o = np.flatnonzero(keep)[np.argsort(ED[keep], kind="stable")][: mv["reserve"]]
    closest = [(int(EI[k]), int(EJ[k]), float(ED[k])) for k in o]

    rng = np.random.RandomState(cfg["seed"] + 7)
    allowed = [i for i, w in enumerate(words) if w not in excl]
    by_len = {}
    for i in allowed:
        by_len.setdefault(len(words[i]), []).append(i)
    prof = list(Counter(tuple(sorted((len(words[a]), len(words[b])))) for a, b, _ in closest)
                .elements())
    rng.shuffle(prof)
    randoms, seen = [], set()
    for la, lb in prof:
        while True:
            a, b = int(rng.choice(by_len[la])), int(rng.choice(by_len[lb]))
            k = tuple(sorted((a, b)))
            if a != b and k not in seen:
                seen.add(k)
                randoms.append((k[0], k[1], float("nan")))
                break

    reserve_words = sorted({words[i] for p in closest + randoms for i in p[:2]})
    chans = {k: build_channel(cfg, k, C.stage(cfg, "cache")) for k in ("original", "heldout")}
    if args.recover_from_cache:
        # GPU-free recovery: the gate is not cached, but its outcome is: the
        # evaluated words are exactly the spoken words of the cached original-
        # channel trials, and every one of them passed the gate on both
        # channels.  take() keeps the first n_pairs pairs whose words both
        # passed, so selecting the first n_pairs pairs with both words in that
        # set reproduces the selection exactly (counts are checked below).
        ch = chans["original"]
        pre = f"{ch.tts.name}|{ch.scorer.name}|"
        suf = f"|{cfg['seed'] + 7}"
        keys = [r[0] for r in ch.db.execute("SELECT k FROM fc WHERE k LIKE ?", (pre + "%",))
                if r[0].endswith(suf)]
        passed = {k.split("|")[3] for k in keys}
        ok = {k: {w: w in passed for w in reserve_words} for k in chans}
        print(f"recovered gate outcome from cache: {len(passed)} words passed", flush=True)
    else:
        # gate against 50 random distractors from outside every reserve pair, so a
        # word never competes with its own pair partner (that would remove exactly
        # the confusable pairs this test is about)
        drng = np.random.RandomState(cfg["seed"] + 8)
        rs = set(reserve_words)
        pool = [words[i] for i in allowed if words[i] not in rs]
        distractors = [pool[i] for i in drng.choice(len(pool), 50, replace=False)]
        ok = {k: ch.gate(reserve_words, distractors) for k, ch in chans.items()}
    good = lambda i: all(ok[k][words[i]] for k in chans)

    def take(lst):
        sel, skipped = [], 0
        for a, b, d in lst:
            if good(a) and good(b):
                sel.append((a, b, d))
                if len(sel) == mv["n_pairs"]:
                    break
            else:
                skipped += 1
        return sel, skipped

    sel_c, sk_c = take(closest)
    sel_r, sk_r = take(randoms)
    eval_words = sorted({words[i] for p in sel_c + sel_r for i in p[:2]})
    print(f"{len(sel_c)} closest / {len(sel_r)} random pairs; skipped {sk_c}/{sk_r}; "
          f"{len(eval_words)} words", flush=True)
    if args.recover_from_cache:
        # must match the Colab run exactly (100/100 pairs, 391 words, same hash)
        assert set(eval_words) == passed, "recovered selection differs from the cached run"
        assert (sk_c, sk_r) == (29, 24), f"skips {sk_c}/{sk_r} differ from the Colab run (29/24)"

    res = {"decision_rule": rule, "skipped_closest": sk_c, "skipped_random": sk_r,
           "self_recognition_failures": {k: [w for w, v in d.items() if not v] for k, d in ok.items()}}
    if args.recover_from_cache:
        # per-channel failure lists are not recoverable without the GPU gate
        res["self_recognition_failures"] = None
        res["note"] = ("gate outcome and original-channel trials recovered from the Colab "
                       "trial cache; held-out trials partly cached, rest scored on CPU")
    rows = []
    for k, ch in chans.items():
        tr = ch.run(eval_words, eval_words, mv["snr_db"], mv["trials"], seed=cfg["seed"] + 7)
        tr.to_csv(out / f"trials_{k}.csv", index=False, encoding="utf-8-sig")
        g = {w: d for w, d in tr.groupby("true_word")}

        def rate(a, b):
            wa, wb = words[a], words[b]
            n = len(g[wa]) + len(g[wb])
            c = int((g[wa][rule] == wb).sum()) + int((g[wb][rule] == wa).sum())
            return c / n
        rc = np.array([rate(a, b) for a, b, _ in sel_c])
        rr = np.array([rate(a, b) for a, b, _ in sel_r])
        res[k] = separation(rc, rr, 2000, cfg["seed"])
        res[k]["error_rate"] = float((tr.true_word != tr[rule]).mean())
        for grp, sel, rates in (("closest", sel_c, rc), ("random", sel_r, rr)):
            for (a, b, d), x in zip(sel, rates):
                rows.append((k, grp, words[a], words[b], d, x))
        print(k, {kk: v for kk, v in res[k].items()}, flush=True)
    pd.DataFrame(rows, columns=["channel", "group", "w1", "w2", "d", "rate"]).to_csv(
        out / "pair_rates.csv", index=False, encoding="utf-8-sig")
    C.save(res, out / "result.json")
    C.manifest(cfg, "metric_validation", {"n_words": len(eval_words)})


if __name__ == "__main__":
    main()
