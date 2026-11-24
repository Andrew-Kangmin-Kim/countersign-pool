"""Step 8 — metric selection: which distance best predicts confusion?
ORIGINAL channel only (edge-tts -> Whisper); the held-out channel is not used,
so it stays held out.  Run after `s2_calibration.py --section calibration_v2`.

Metrics (config metric_selection.metrics):
  calibrated_edit  weighted Manhattan edit distance, weights from calibration_v2
  manhattan / euclidean / cosine
                   on fixed-length syllable-slot vectors (cpool/vectors.py)

Design — one shared pair set, every metric scored on the same pairs:
  * near range of a metric = its near_k nearest pairs (ties at the boundary
    included).  per_metric pairs are drawn uniformly at random from each
    metric's near range (random tie-breaking, not frequency order), plus
    `random` pairs from the whole universe; duplicates merged.
  * "closest 100" was dropped: 3,047 pairs tie at the Manhattan/Euclidean
    minimum and 487 at the calibrated-edit minimum, so a closest-k list is
    chosen by the tie-break, not by the metric.
  * excluded: calibration_v2 + pilot words (the calibrated metric is scored on
    words it was not fit on) and identical-surface pairs (distance 0 under
    every metric).
  * gate: clean forced choice vs 50 distractors from outside the reserve; a
    failed word removes its pair, the next reserve pair of that stratum replaces it.
  * trials: all evaluated words in ONE candidate set, SNR 0 / -5 dB.
    pair rate = (#a heard as b + #b heard as a) / (#a spoken + #b spoken)
  * score: Spearman(distance, pair rate) over ALL pooled pairs (primary) and
    over the near pairs only; bootstrap over pairs.  Winner = most negative
    primary rho; each metric's rho minus the winner's has a bootstrap CI.

  --plan   build and save the pair set only (no audio, no GPU)
Outputs runs/metric_selection/: pairs.csv, trials.csv, pair_rates.csv, result.json
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cpool import config as C
from cpool import phonetics, vectors
from cpool.phonetics import Params


def edit_near(surfs, P, k, ok, group, tau):
    E = phonetics.Encoded(surfs)
    while True:
        EI, EJ, ED, _ = phonetics.edge_list(E, P, tau, verbose=False)
        m = ok[EI] & ok[EJ] & (group[EI] != group[EJ])
        if m.sum() >= k:
            break
        tau *= 2
    EI, EJ, ED = EI[m], EJ[m], ED[m]
    o = np.lexsort((EJ, EI, ED))
    kth = ED[o[k - 1]]
    o = o[ED[o] <= kth]                                  # ties at the boundary included
    return [(int(EI[t]), int(EJ[t]), float(ED[t])) for t in o]


def rho(d, r):
    v = spearmanr(d, r).statistic
    return float(v) if np.isfinite(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    # --channel heldout: the SAME design scored on the held-out channel, run only
    # AFTER the metric was chosen on the original channel (confirmation, not selection)
    ap.add_argument("--channel", default="original", choices=["original", "heldout"])
    args = ap.parse_args()
    cfg = C.load()
    ms = cfg["metric_selection"]
    seed = cfg["seed"]
    out = C.stage(cfg, "metric_selection" if args.channel == "original" else "metric_selection_heldout")
    uni = pd.read_csv(C.stage(cfg, "lexicon") / "universe.csv", encoding="utf-8-sig")
    words, surfs = uni["lemma"].tolist(), uni["surface"].tolist()
    n = len(words)

    calv2 = C.stage(cfg, "calibration_v2")
    excl = set(C.load_json(calv2 / "words.json")["sample"])
    pw = C.stage(cfg, "pilot_heldout") / "words.json"
    if pw.exists():
        excl |= set(C.load_json(pw)["pilot_words"])
    ok = np.array([w not in excl for w in words])
    group = pd.factorize(pd.Series(surfs))[0]
    n_homophone = int(sum(c * (c - 1) // 2 for c in Counter(surfs).values()))
    P = Params.from_dict(C.load_json(calv2 / "params.json"))
    rule = C.load_json(calv2 / "result.json")["decision_rule"]
    print(f"universe {n:,}; excluded words {int((~ok).sum())}; identical-surface pairs "
          f"excluded {n_homophone}; calibrated params {P.to_dict()}; rule {rule}", flush=True)

    V = vectors.encode(surfs)

    def dist(a, b, m):
        if m == "calibrated_edit":
            return float(phonetics.cross_distances(phonetics.Encoded([surfs[a]]),
                                                   phonetics.Encoded([surfs[b]]), P)[0, 0])
        return vectors.pair_distance(V[a], V[b], m)

    # --- strata: each metric's near range, sampled at random; + random pairs
    rng = np.random.RandomState(seed + 11)
    n_res = ms["per_metric"] * ms["oversample"]
    strata, near_info = {}, {}
    for m in ms["metrics"]:
        near = (edit_near(surfs, P, ms["near_k"], ok, group, ms["edit_tau"])
                if m == "calibrated_edit" else vectors.closest_pairs(V, m, ms["near_k"], ok, group, include_ties=True))
        near_info[m] = {"size": len(near), "d_min": near[0][2], "d_max": near[-1][2]}
        pick = rng.choice(len(near), n_res, replace=False)
        strata[m] = [near[t][:2] for t in pick]
        print(f"{m}: near range {len(near):,} pairs, d {near[0][2]:.3f}..{near[-1][2]:.3f}", flush=True)
    allowed = np.flatnonzero(ok)
    randoms, seen = [], set()
    while len(randoms) < ms["random"] * ms["oversample"]:
        a, b = (int(x) for x in rng.choice(allowed, 2, replace=False))
        k = (min(a, b), max(a, b))
        if group[a] != group[b] and k not in seen:
            seen.add(k)
            randoms.append(k)
    strata["random"] = randoms
    for g in strata:
        strata[g] = [(min(a, b), max(a, b)) for a, b in strata[g]]

    rows = []
    for g, L in strata.items():
        for r, (a, b) in enumerate(L):
            row = {"stratum": g, "rank": r, "w1": words[a], "w2": words[b], "s1": surfs[a], "s2": surfs[b]}
            for m in ms["metrics"]:
                row[f"d_{m}"] = dist(a, b, m)
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "pairs.csv", index=False, encoding="utf-8-sig")
    reserve_words = sorted({words[i] for L in strata.values() for p in L for i in p})
    print(f"reserve words to gate: {len(reserve_words)}", flush=True)
    if args.plan:
        return

    from cpool.channel import build_channel
    ch = build_channel(cfg, args.channel, C.stage(cfg, "cache"))
    drng = np.random.RandomState(seed + 12)
    rs = set(reserve_words)
    pool = [words[i] for i in allowed if words[i] not in rs]
    distractors = [pool[i] for i in drng.choice(len(pool), 50, replace=False)]
    gate = ch.gate(reserve_words, distractors)

    want = {g: (ms["random"] if g == "random" else ms["per_metric"]) for g in strata}
    sel, skipped, taken = {}, {}, set()
    for g, L in strata.items():
        s, sk = [], 0
        for a, b in L:
            if len(s) == want[g]:
                break
            if not (gate[words[a]] and gate[words[b]]):
                sk += 1
            elif (a, b) not in taken:                    # a pair drawn by two strata counts once
                s.append((a, b))
                taken.add((a, b))
        sel[g], skipped[g] = s, sk
    pairs = [(g, a, b) for g, s in sel.items() for a, b in s]
    eval_words = sorted({words[i] for _, a, b in pairs for i in (a, b)})
    print({g: len(s) for g, s in sel.items()}, "skipped", skipped,
          f"; {len(pairs)} pairs, {len(eval_words)} words", flush=True)

    tr = ch.run(eval_words, eval_words, ms["snr_db"], ms["trials"], seed=seed + 13)
    tr.to_csv(out / "trials.csv", index=False, encoding="utf-8-sig")
    by = {w: d for w, d in tr.groupby("true_word")}

    def rate(a, b, snr=None):
        ga, gb = by[words[a]], by[words[b]]
        if snr is not None:
            ga, gb = ga[ga.snr == snr], gb[gb.snr == snr]
        c = int((ga[rule] == words[b]).sum()) + int((gb[rule] == words[a]).sum())
        return c / (len(ga) + len(gb))

    df = pd.DataFrame([{"stratum": g, "w1": words[a], "w2": words[b],
                        **{f"d_{m}": dist(a, b, m) for m in ms["metrics"]},
                        "rate": rate(a, b),
                        **{f"rate_{s}dB": rate(a, b, s) for s in ms["snr_db"]}}
                       for g, a, b in pairs])
    df.to_csv(out / "pair_rates.csv", index=False, encoding="utf-8-sig")

    near = (df.stratum != "random").to_numpy()
    res = {"decision_rule": rule, "excluded_words": int((~ok).sum()),
           "identical_surface_pairs_excluded": n_homophone, "near_ranges": near_info,
           "skipped": skipped, "n_pairs": {g: len(s) for g, s in sel.items()},
           "n_eval_words": len(eval_words),
           "pairs_with_confusion": int((df.rate > 0).sum()),
           "error_rate": float((tr.true_word != tr[rule]).mean()),
           "error_by_snr": {str(k): float(v) for k, v in
                            (tr.true_word != tr[rule]).groupby(tr.snr).mean().items()},
           "metrics": {}}
    B = ms["bootstrap"]
    brng = np.random.RandomState(seed + 14)
    boot = {m: [] for m in ms["metrics"]}
    boot_near = {m: [] for m in ms["metrics"]}
    nidx = np.flatnonzero(near)
    for _ in range(B):
        i = brng.randint(len(df), size=len(df))
        j = nidx[brng.randint(len(nidx), size=len(nidx))]
        for m in ms["metrics"]:
            boot[m].append(rho(df[f"d_{m}"].values[i], df.rate.values[i]))
            boot_near[m].append(rho(df[f"d_{m}"].values[j], df.rate.values[j]))
    for m in ms["metrics"]:
        d = df[f"d_{m}"].values
        res["metrics"][m] = {
            "spearman_all": rho(d, df.rate.values),
            "spearman_all_ci": [float(np.nanpercentile(boot[m], 2.5)), float(np.nanpercentile(boot[m], 97.5))],
            "spearman_near": rho(d[near], df.rate.values[near]),
            "spearman_near_ci": [float(np.nanpercentile(boot_near[m], 2.5)),
                                 float(np.nanpercentile(boot_near[m], 97.5))],
            "spearman_all_by_snr": {str(s): rho(d, df[f"rate_{s}dB"].values) for s in ms["snr_db"]},
        }
    win = min(ms["metrics"], key=lambda m: res["metrics"][m]["spearman_all"])
    for m in ms["metrics"]:
        diff = np.array(boot[m]) - np.array(boot[win])
        res["metrics"][m]["minus_winner_ci"] = [float(np.nanpercentile(diff, 2.5)),
                                                float(np.nanpercentile(diff, 97.5))]
    res["winner"] = win
    C.save(res, out / "result.json")
    res["channel"] = args.channel
    C.save(res, out / "result.json")
    C.manifest(cfg, out.name, {"winner": win, "tts": ch.tts.name, "asr": ch.scorer.name})
    for m in ms["metrics"]:
        r = res["metrics"][m]
        print(f"{m:16s} rho_all={r['spearman_all']:.3f} {np.round(r['spearman_all_ci'], 3).tolist()}  "
              f"rho_near={r['spearman_near']:.3f} {np.round(r['spearman_near_ci'], 3).tolist()}  "
              f"minus winner {np.round(r['minus_winner_ci'], 3).tolist()}", flush=True)
    print("winner:", win)


if __name__ == "__main__":
    main()
