"""Step 6 — pool validation on the HELD-OUT channel.

For each x in {5, 10, 50}: the optimized pool at tau*_x vs three baselines
  random         N words drawn at random from the x candidate set
  top_freq       the N most frequent candidates
  greedy_noswap  min-degree greedy without swaps (its own N)
The listener chooses among the WHOLE pool; `spoken` words per pool are sampled
(same seeded rule for every pool).  Metric: in-pool confusion rate per SNR,
bootstrap CI over spoken words.  Clean self-recognition is reported, not used
to drop words.  Words the ASR cannot represent are not spoken (count reported).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from cpool import config as C
from cpool.channel import build_channel
from cpool.stats import boot_mean_ci


def main():
    cfg = C.load()
    pv = cfg["pool_validation"]
    out = C.stage(cfg, "pool_validation")
    uni = pd.read_csv(C.stage(cfg, "lexicon") / "universe.csv", encoding="utf-8-sig")
    words = uni["lemma"].tolist()
    n_x = {float(k): v for k, v in C.load_json(C.stage(cfg, "lexicon") / "summary.json")["N_x"].items()}
    rule = C.load_json(C.stage(cfg, "calibration") / "result.json")["decision_rule"]
    pdir = C.stage(cfg, "pools")
    ch = build_channel(cfg, "heldout", C.stage(cfg, "cache"))

    pools = {}
    for x in sorted(n_x):
        f = pdir / f"optimized_x{x:g}.csv"
        if not f.exists():
            continue
        opt = pd.read_csv(f, encoding="utf-8-sig")["word"].tolist()
        N = len(opt)
        rng = np.random.RandomState(cfg["seed"] + int(x))
        pools[f"x{x:g}/optimized"] = opt
        pools[f"x{x:g}/random"] = [words[i] for i in sorted(rng.choice(n_x[x], N, replace=False))]
        pools[f"x{x:g}/top_freq"] = words[:N]
        pools[f"x{x:g}/greedy_noswap"] = pd.read_csv(pdir / f"greedy_noswap_x{x:g}.csv",
                                                     encoding="utf-8-sig")["word"].tolist()

    res, per_word = {}, []
    for name, pw in pools.items():
        srng = np.random.RandomState(cfg["seed"] + 40)
        # words the held-out ASR cannot write can never be recognised; they stay
        # in the candidate set but are not spoken (count reported, agreed rule)
        speakable = [w for w in pw if ch.scorer.representable(w)]
        k = min(pv["spoken"], len(speakable))
        spoken = [speakable[i] for i in sorted(srng.choice(len(speakable), k, replace=False))]
        clean = ch.self_recognition(spoken, candidates=pw)
        frames = []
        for s in pv["seeds"]:
            t = ch.run(spoken, pw, pv["snr_db"], pv["trials"], seed=cfg["seed"] + 100 + s)
            frames.append(t.assign(seed=s))
        tr = pd.concat(frames)
        tr["err"] = (tr.true_word != tr[rule]).astype(float)
        tr.to_csv(out / f"trials_{name.replace('/', '_')}.csv", index=False, encoding="utf-8-sig")
        r = {"N": len(pw), "bits": float(np.log2(len(pw))), "spoken": k,
             "n_unrepresentable": len(pw) - len(speakable),
             "clean_self_recognition": float(np.mean(list(clean.values())))}
        for snr, g in tr.groupby("snr"):
            pwm = g.groupby("true_word")["err"].mean()
            m, ci = boot_mean_ci(pwm.to_numpy(), 2000, cfg["seed"])
            r[f"conf_{snr}dB"], r[f"conf_{snr}dB_ci"] = m, ci
            per_word += [(name, snr, w, v) for w, v in pwm.items()]
        res[name] = r
        print(name, r, flush=True)
        C.save(res, out / "result.json")
    pd.DataFrame(per_word, columns=["pool", "snr", "word", "err"]).to_csv(
        out / "per_word.csv", index=False, encoding="utf-8-sig")
    C.manifest(cfg, "pool_validation", {"pools": list(pools), "tts": ch.tts.name,
                                        "asr": ch.scorer.name})


if __name__ == "__main__":
    main()
