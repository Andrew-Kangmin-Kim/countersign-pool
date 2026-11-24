"""Pick the held-out channel: gTTS -> a non-Whisper CTC ASR.

Criteria (fixed before running):
  1. shares no model with the original channel (edge-tts / Whisper)  -- all wav2vec2
  2. can represent 100% of the x=5% universe words
  3. clean self-recognition >= 95% on the pilot words (gTTS audio)
  4. prior-corrected error at the metric-validation SNRs in the informative
     band [0.3, 0.7] (so both channels use the same SNRs)
Pilot words: 40 random universe words from the END of a seeded permutation
(calibration and metric validation draw from the front / by distance and
exclude these).  Writes runs/pilot_heldout/table.csv and choice.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from cpool import config as C
from cpool.channel import CTCScorer, Channel, GoogleTTS, SenseVoiceScorer

CANDIDATES = ["kresnik/wav2vec2-large-xlsr-korean", "Hyuk/wav2vec2-korean-v2",
              "Kkonjeong/wav2vec2-base-korean", "dgkim0306/wav2vec2-large-xls-r-300m-korean-g",
              "sensevoice:iic/SenseVoiceSmall"]
SNRS = [10, 5, 0, -5, -10]


def main():
    cfg = C.load()
    out = C.stage(cfg, "pilot_heldout")
    uni = pd.read_csv(C.stage(cfg, "lexicon") / "universe.csv", encoding="utf-8-sig")
    perm = np.random.RandomState(cfg["seed"] + 999).permutation(len(uni))
    pilot = uni["lemma"].iloc[perm[-40:]].tolist()
    C.save({"pilot_words": pilot}, out / "words.json")
    cache = C.stage(cfg, "cache")
    tts = GoogleTTS(cache / "tts")
    rows = []
    import sys as _s
    only = _s.argv[1:]
    prev = out / "table.csv"
    rows = pd.read_csv(prev).to_dict("records") if prev.exists() and only else []
    rows = [r for r in rows if r["model"] not in only]
    for mid in (only or CANDIDATES):
        sc = SenseVoiceScorer(mid.split(":", 1)[1], "cpu", 512) if mid.startswith("sensevoice:")             else CTCScorer(mid, "cpu", 512)
        cover = float(np.mean([sc.representable(w) for w in uni["lemma"]]))
        words = [w for w in pilot if sc.representable(w)]
        ch = Channel("pilot", tts, sc, cache / "trials.sqlite")
        ok = ch.self_recognition(words)
        tr = ch.run(words, words, SNRS, 3, seed=cfg["seed"] + 999, log_every=10 ** 9)
        err = (tr.true_word != tr.pred_corr).groupby(tr.snr).mean()
        mv = [err.get(s, np.nan) for s in cfg["metric_validation"]["snr_db"]]
        row = {"model": mid, "coverage": cover, "clean_self_recognition": float(np.mean(list(ok.values()))),
               **{f"err_{s}dB": float(err.get(s, np.nan)) for s in SNRS}}
        row["c2_coverage"] = cover == 1.0
        row["c3_clean"] = row["clean_self_recognition"] >= 0.95
        row["c4_informative"] = all(0.3 <= e <= 0.7 for e in mv)
        row["passes"] = row["c2_coverage"] and row["c3_clean"] and row["c4_informative"]
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        del sc
    df = pd.DataFrame(rows)
    df.to_csv(out / "table.csv", index=False)
    passing = df[df.passes]
    choice = passing.iloc[0]["model"] if len(passing) else None
    C.save({"choice": choice, "criteria_met_by": passing["model"].tolist()}, out / "choice.json")
    print("CHOICE:", choice)


if __name__ == "__main__":
    main()
