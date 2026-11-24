"""Step 2 — calibration on the ORIGINAL channel (needs a GPU: run on Colab).

Notebook procedure (Cells 7-9) on the x=5% universe; see cpool/calibration.py
for the two bug fixes.  Outputs runs/calibration/: params.json, result.json,
words.json, trials.csv, pairs.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cpool import calibration as cal
from cpool import config as C
from cpool import phonetics
from cpool.channel import build_channel
from cpool.phonetics import Params


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # --section calibration_v2: the higher-signal refit of the metric-selection
    # experiment; it writes runs/calibration_v2/ and leaves runs/calibration/ alone
    ap.add_argument("--section", default="calibration")
    args = ap.parse_args()
    cfg = C.load()
    c = cfg[args.section]
    out = C.stage(cfg, args.section)
    uni = pd.read_csv(C.stage(cfg, "lexicon") / "universe.csv", encoding="utf-8-sig")
    pilot = set(C.load_json(C.stage(cfg, "pilot_heldout") / "words.json")["pilot_words"]) \
        if (C.stage(cfg, "pilot_heldout") / "words.json").exists() else set()
    words = uni["lemma"].tolist()
    surf = dict(zip(uni["lemma"], uni["surface"]))
    allowed = np.array([i for i, w in enumerate(words) if w not in pilot])

    # --- sample (notebook Cell 7): scan 800, 30 closest pairs, rest random from the scan
    rng = np.random.RandomState(cfg["seed"])
    scan = rng.choice(allowed, size=c["scan_words"], replace=False)
    U = Params(np.ones(4), np.ones(4), cfg["phonetics"]["gap"], cfg["phonetics"]["cross"])
    E = phonetics.Encoded([surf[words[i]] for i in scan], cfg["phonetics"]["maxlen"])
    D = phonetics.cross_distances(E, E, U)
    iu = np.triu_indices(len(scan), 1)
    order = np.argsort(D[iu], kind="stable")[: c["near_pairs"]]
    close = {int(scan[iu[0][k]]) for k in order} | {int(scan[iu[1][k]]) for k in order}
    rest = [i for i in scan if i not in close]
    idx = sorted(close) + list(rng.choice(rest, size=max(0, c["n_words"] - len(close)), replace=False))
    sample = [words[i] for i in idx]

    ch = build_channel(cfg, "original", C.stage(cfg, "cache"))
    # gate: can TTS->ASR say/hear the word? tested against 50 random distractors
    # from OUTSIDE the sample (never against its near-pair partner)
    drng = np.random.RandomState(cfg["seed"] + 5)
    pool = [words[i] for i in allowed if words[i] not in set(sample)]
    distractors = [pool[i] for i in drng.choice(len(pool), 50, replace=False)]
    ok = ch.gate(sample, distractors)
    kept = [w for w in sample if ok[w]]
    print(f"calibration sample {len(sample)} ({len(close)} from near pairs); "
          f"self-recognized {len(kept)}", flush=True)
    C.save({"sample": sample, "near": [words[i] for i in sorted(close)], "kept": kept},
           out / "words.json")

    tr = ch.run(kept, kept, c["snr_db"], c["trials"], seed=cfg["seed"])
    tr.to_csv(out / "trials.csv", index=False, encoding="utf-8-sig")
    err = (tr.true_word != tr.pred_raw).groupby(tr.snr).mean()
    print("error by SNR (raw):", err.round(3).to_dict(), flush=True)

    # --- decision rule (notebook Cell 8): the one the uniform distance predicts better
    rules = {}
    for col in ("pred_raw", "pred_corr"):
        p = cal.pair_table(tr, kept, col)
        rules[col] = (spearmanr(cal.distances(p, surf, U), p.rate).statistic, p)
    rule = min(rules, key=lambda k: rules[k][0])
    pairs = rules[rule][1]
    print("decision rule:", rule, {k: round(v[0], 4) for k, v in rules.items()}, flush=True)

    res = cal.calibrate(pairs, surf, cfg, args.section)
    P = res.pop("params")
    pairs.assign(d=cal.distances(pairs, surf, P)).to_csv(out / "pairs.csv", index=False,
                                                         encoding="utf-8-sig")
    (out / "params.json").write_text(json.dumps(P.to_dict(), indent=2), encoding="utf-8")
    res.update({"decision_rule": rule,
                "rule_spearman": {k: float(v[0]) for k, v in rules.items()},
                "error_by_snr_raw": {str(k): float(v) for k, v in err.items()},
                "n_pairs": len(pairs), "pairs_with_confusion": int((pairs.confusions > 0).sum())})
    C.save(res, out / "result.json")
    C.manifest(cfg, args.section, {"adopted": res["adopted"], "params": P.to_dict(),
                                    "tts": ch.tts.name, "asr": ch.scorer.name})
    print("adopted:", res["adopted"], res["spearman"], P.to_dict())


if __name__ == "__main__":
    main()
