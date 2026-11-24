"""Step 3 — distances ONCE on the x=5% universe, edges with d <= tau_max.
Metric from config distance.metric (calibrated_edit, or a vector metric from s8).
Outputs runs/distances/edges.npz (i, j, d) + summary.json."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd

from cpool import config as C
from cpool import phonetics, vectors
from cpool.phonetics import Params


def main():
    cfg = C.load()
    out = C.stage(cfg, "distances")
    uni = pd.read_csv(C.stage(cfg, "lexicon") / "universe.csv", encoding="utf-8-sig")
    tau_max = float(cfg["grid"]["tau"][1])
    metric = cfg.get("distance", {}).get("metric", "calibrated_edit")
    if metric != "calibrated_edit":
        # vector metric chosen in s8 (metric selection): edges with d <= tau_max
        V = vectors.encode(uni["surface"].tolist())
        t0 = time.time()
        EI, EJ, ED = vectors.edge_list(V, metric, tau_max)
        info = {"metric": metric, "n": len(V), "tau_max": tau_max, "edges": int(len(ED)),
                "seconds": time.time() - t0}
        np.savez_compressed(out / "edges.npz", i=EI, j=EJ, d=ED)
        C.save(info, out / "summary.json")
        C.manifest(cfg, "distances", info)
        print(info)
        return
    P = Params.from_dict(C.load_json(C.stage(cfg, "calibration") / "params.json"))
    assert tau_max <= 2 * P.gap, "length blocking needs tau_max <= 2*gap"
    E = phonetics.Encoded(uni["surface"].tolist(), cfg["phonetics"]["maxlen"])
    t0 = time.time()
    EI, EJ, ED, info = phonetics.edge_list(E, P, tau_max + 1e-6)
    info.update({"seconds": time.time() - t0, "edges": int(len(ED)), "params": P.to_dict()})
    np.savez_compressed(out / "edges.npz", i=EI, j=EJ, d=ED)
    C.save(info, out / "summary.json")
    C.manifest(cfg, "distances", info)
    print(info)


if __name__ == "__main__":
    main()
