"""Step 5 — one graph per (x, tau) cell, solver per cell, Kneedle knee of N(tau) per x,
final pools at (x, tau*_x) with the frequency tie-break.

Per cell (cpool.graph.solve_cell): ILP feasibility test on the giant
component -> exact alpha if it works, else solver upper bound as reference;
greedy (+swaps), forest and no-swap greedy always run, gaps vs the reference.
Outputs: runs/grid/grid.csv (resumable), knee.json, pools/*.csv

  --estimate   time a few cells per x with a short ILP limit and print an
               expected total, without writing results
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from cpool import config as C
from cpool import graph


def taus(cfg):
    a, b, s = cfg["grid"]["tau"]
    return np.round(np.arange(a, b + 1e-9, s), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()
    cfg = C.load()
    sv = cfg["solver"]
    out = C.stage(cfg, "grid")
    uni = pd.read_csv(C.stage(cfg, "lexicon") / "universe.csv", encoding="utf-8-sig")
    n_x = {float(k): v for k, v in C.load_json(C.stage(cfg, "lexicon") / "summary.json")["N_x"].items()}
    z = np.load(C.stage(cfg, "distances") / "edges.npz")
    EI, EJ, ED = z["i"], z["j"], z["d"]
    T = taus(cfg)
    xs = sorted(n_x)
    # grid cells: short ILP on the giant (bound for the curve); the full
    # ilp_time_limit_s test runs only at each x's knee cell
    sv_grid = dict(sv, ilp_time_limit_s=sv["grid_ilp_time_limit_s"])

    if a.estimate:
        for x in xs:
            for tau in (T[len(T) // 4], T[len(T) // 2], T[-1]):
                G = graph.Graph.from_edges(EI, EJ, ED, n_x[x], float(tau))
                r = graph.solve_cell(G, sv_grid, seed=cfg["seed"])
                print(f"x={x:g} tau={tau}: n={r['n']} E={r['edges']} giant={r['giant']} "
                      f"ILP={r['giant_ilp_status']} heuristics "
                      f"{r['sec_greedy']+r['sec_forest']:.0f}s ilp {r['sec_ilp']:.0f}s "
                      f"cell {r['sec_cell']:.0f}s", flush=True)
        return

    gp = out / "grid.csv"
    done = pd.read_csv(gp) if gp.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    have = {(r["x"], round(r["tau"], 4)) for r in rows}
    for x in xs:
        for tau in T:                                   # sparse -> dense
            if (x, round(float(tau), 4)) in have:
                continue
            G = graph.Graph.from_edges(EI, EJ, ED, n_x[x], float(tau))
            r = graph.solve_cell(G, sv_grid, seed=cfg["seed"])
            row = {"x": x, "tau": float(tau), **{k: v for k, v in r.items() if not k.startswith("_")}}
            rows.append(row)
            pd.DataFrame(rows).to_csv(gp, index=False)
            print(f"x={x:g} tau={tau:.2f} n={r['n']} E={r['edges']} comps={r['components']} "
                  f"giant={r['giant']} ILP={r.get('giant_ilp_status', 'none_needed')} "
                  f"{'alpha=' if r['alpha_exact'] else 'UB='}{r['reference']} N={r['N']} "
                  f"bits={r['bits']:.3f} gap={r['gap_words']} | greedy {r['N_greedy']} "
                  f"forest {r['N_forest']} noswap {r['N_greedy_noswap']} ({r['sec_cell']:.0f}s)",
                  flush=True)

    grid = pd.DataFrame(rows)
    knees = {}
    pools = C.stage(cfg, "pools")
    for x in xs:
        g = grid[grid.x == x].sort_values("tau")
        t_star, curve = graph.knee(g.tau.values, g.N.values)
        knees[str(x)] = {"tau_star": t_star, "curve": curve}
        if t_star is None:
            print(f"x={x:g}: Kneedle found no knee")
            continue
        n = n_x[x]
        G = graph.Graph.from_edges(EI, EJ, ED, n, t_star)
        print(f"x={x:g}: knee tau*={t_star}; full ILP test ({sv['ilp_time_limit_s']} s) "
              f"on the giant component", flush=True)
        r = graph.solve_cell(G, sv, seed=cfg["seed"])
        S = graph.lexicographic_pool(G, r["_best"], sv, {v: n - v for v in range(n)},
                                     seed=cfg["seed"])
        assert len(S) == r["N"] and G.is_independent(S)
        knees[str(x)].update({"N": len(S), "bits": math.log2(len(S)),
                              "knee_cell": {k: v for k, v in r.items() if not k.startswith("_")}})
        pd.DataFrame({"word": uni["lemma"].iloc[S].values, "index": S,
                      "zipf": uni["zipf"].iloc[S].values}).to_csv(
            pools / f"optimized_x{x:g}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"word": uni["lemma"].iloc[r["_greedy_noswap"]].values,
                      "index": r["_greedy_noswap"]}).to_csv(
            pools / f"greedy_noswap_x{x:g}.csv", index=False, encoding="utf-8-sig")
        print(f"pool x={x:g}: tau*={t_star} N={len(S)} bits={math.log2(len(S)):.3f}")
    C.save(knees, out / "knee.json")
    C.manifest(cfg, "grid", {"cells": len(grid), "knees": knees})


if __name__ == "__main__":
    main()
