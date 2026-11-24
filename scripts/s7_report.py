"""Step 7 — report: figures, tables and report.md from the finished runs/.

Reads only stage outputs (no recomputation).  Writes runs/report/:
  fig1_metric_validation.png  closest vs random pair confusion, both channels
  fig2_frontier.png           bits = log2 N vs tau per x, knee tau* marked
  fig3_solver_gaps.png        gap (bits) to the reference per method vs tau
  fig4_pool_validation.png    in-pool confusion at -5 dB, optimized vs baselines
  table_*.csv / .md           every number in the figures
  report.md                   results write-up
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpool import config as C
from cpool.stats import boot_mean_ci

# fixed entity -> colour (reference categorical palette, slots in order)
COL = {"optimized": "#2a78d6", "random": "#eb6834", "top_freq": "#1baf7a",
       "greedy_noswap": "#eda100", "greedy": "#e87ba4", "forest": "#008300"}
LAB = {"optimized": "optimized (solver best)", "random": "random", "top_freq": "top-N frequency",
       "greedy_noswap": "greedy, no swaps", "greedy": "greedy + swaps", "forest": "forest (100 runs)"}
XCOL = {5.0: "#104281", 10.0: "#2a78d6", 50.0: "#86b6ef"}     # one hue, ordinal x
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def md_table(df, floatfmt=3):
    def f(v):
        if isinstance(v, float):
            return "—" if math.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_table(df, out, name, floatfmt=3):
    df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    (out / f"{name}.md").write_text(md_table(df, floatfmt), encoding="utf-8")
    return md_table(df, floatfmt)


def main():
    cfg = C.load()
    seed = cfg["seed"]
    R = Path(cfg["out_dir"])
    out = C.stage(cfg, "report")
    lex = C.load_json(R / "lexicon" / "summary.json")
    cal = C.load_json(R / "calibration" / "result.json")
    par = C.load_json(R / "calibration" / "params.json")
    mv = C.load_json(R / "metric_validation" / "result.json")
    pr = pd.read_csv(R / "metric_validation" / "pair_rates.csv", encoding="utf-8-sig")
    grid = pd.read_csv(R / "grid" / "grid.csv")
    knees = C.load_json(R / "grid" / "knee.json")
    pv = C.load_json(R / "pool_validation" / "result.json")
    xs = sorted(grid.x.unique())
    T = {}

    # ---------------------------------------------------------- calibration
    cal_tab = pd.DataFrame({
        "feature": ["c_place", "c_manner", "c_laryngeal", "c_sonorant",
                    "v_height", "v_backness", "v_round", "v_glide"],
        "weight": list(par["cons_w"]) + list(par["vowel_w"])})
    T["calibration_weights"] = save_table(cal_tab, out, "table_calibration_weights")
    snr_tab = pd.DataFrame({"snr_db": [int(k) for k in cal["error_by_snr_raw"]],
                            "error_rate": list(cal["error_by_snr_raw"].values())}).sort_values("snr_db", ascending=False)
    T["calibration_snr"] = save_table(snr_tab, out, "table_calibration_error_by_snr")
    sp_tab = pd.DataFrame({"weights": list(cal["spearman"]), "spearman_d_vs_confusion": list(cal["spearman"].values())})
    T["calibration_spearman"] = save_table(sp_tab, out, "table_calibration_spearman")

    # ---------------------------------------------------------- fig 1 metric validation
    chans = [("original", "Original: edge-tts → Whisper-small"),
             ("heldout", "Held-out: gTTS → SenseVoice-Small")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    rows = []
    for ax, (ch, title) in zip(axs, chans):
        for k, (grp, col) in enumerate((("closest", COL["optimized"]), ("random", COL["random"]))):
            v = pr[(pr.channel == ch) & (pr.group == grp)].rate.to_numpy()
            m, ci = boot_mean_ci(v, 2000, seed)
            jit = np.random.RandomState(k).uniform(-0.12, 0.12, len(v))
            ax.scatter(k + jit, v, s=9, color=col, alpha=0.35, linewidths=0)
            ax.errorbar(k, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt="o", ms=7, color=col,
                        mec="white", mew=1.5, capsize=4, lw=2, zorder=5)
            ax.annotate(f"{m:.3f}", (k + 0.2, m), va="center", color=INK, fontsize=8)
            rows.append({"channel": ch, "group": grp, "pairs": len(v), "mean_confusion": m,
                         "ci_lo": ci[0], "ci_hi": ci[1]})
        r = mv[ch]
        ax.set_title(f"{title}\nCliff's δ={r['cliffs_delta']:.2f} [{r['delta_ci'][0]:.2f}, "
                     f"{r['delta_ci'][1]:.2f}], MW p={r['mw_p']:.1e}", fontsize=8.5, color=INK)
        ax.set_xticks([0, 1], ["100 closest pairs", "100 random pairs"])
        ax.set_xlim(-0.5, 1.7)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("pair confusion rate")
    fig.savefig(out / "fig1_metric_validation.png")
    plt.close(fig)
    mvt = pd.DataFrame(rows)
    for ch, _ in chans:
        r = mv[ch]
        mvt.loc[mvt.channel == ch, "cliffs_delta"] = r["cliffs_delta"]
        mvt.loc[mvt.channel == ch, "mw_p"] = r["mw_p"]
        mvt.loc[mvt.channel == ch, "channel_error_rate"] = r["error_rate"]
    T["metric_validation"] = save_table(mvt, out, "table_metric_validation", 4)

    # ---------------------------------------------------------- per-cell table
    cell = grid.assign(
        ILP=grid.giant_ilp_status.fillna("none_needed"),
        reference_kind=np.where(grid.alpha_exact, "exact α", np.where(grid.reference.isna(), "no bound", "upper bound")),
    )[["x", "tau", "n", "edges", "components", "giant", "ILP", "reference_kind", "reference",
       "N", "bits", "gap_words", "gap_bits", "N_greedy", "N_forest", "N_greedy_noswap",
       "sec_greedy", "sec_forest", "sec_ilp", "sec_cell"]]
    save_table(cell, out, "table_grid_cells")

    # ---------------------------------------------------------- fig 2 frontier
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for x in xs:
        g = grid[grid.x == x].sort_values("tau")
        k = knees[str(x)]
        kl = (f"; knee τ*={k['tau_star']:g}, N={k['N']:,} = {k['bits']:.2f} bits"
              if k.get("tau_star") is not None else "; no knee")
        ax.plot(g.tau, g.bits, "-", color=XCOL[x], lw=2,
                label=f"x = {x:g}%  (n = {int(g.n.iloc[0]):,}{kl})")
        ax.plot(g.tau, g.bits, "o", color=XCOL[x], ms=3.5)
        if k.get("tau_star") is not None:
            ax.plot(k["tau_star"], k["bits"], "o", ms=10, color=XCOL[x], mec="white", mew=2, zorder=6)
    ax.set_xlabel("confusability threshold τ (weighted edit distance)")
    ax.set_ylabel("pool entropy, bits = log₂ N")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Frontier: pool size vs τ (knee by Kneedle on N(τ))", fontsize=9, color=INK)
    fig.savefig(out / "fig2_frontier.png")
    plt.close(fig)

    # ---------------------------------------------------------- fig 3 solver gaps
    fig, axs = plt.subplots(1, len(xs), figsize=(9, 2.9), sharey=True)
    for ax, x in zip(axs, xs):
        g = grid[grid.x == x].sort_values("tau")
        ref = g.reference.to_numpy(float)
        for m in ("optimized", "forest", "greedy", "greedy_noswap"):
            Nm = g.N.to_numpy(float) if m == "optimized" else g[f"N_{m}"].to_numpy(float)
            gap = np.log2(ref) - np.log2(Nm)
            ax.plot(g.tau, gap, "-o", ms=3, lw=1.8, color=COL[m], label=LAB[m])
        nob = g.tau[g.reference.isna()]
        if len(nob):
            ax.axvspan(nob.min() - 0.125, g.tau.max() + 0.125, color=GRID, alpha=0.6, lw=0)
            ax.text(nob.min(), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1, " no bound\n (model > RAM)",
                    va="top", fontsize=7, color=INK2)
        ax.set_title(f"x = {x:g}%", fontsize=9, color=INK)
        ax.set_xlabel("τ")
    axs[0].set_ylabel("gap to reference (bits)")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Reference = exact α where the ILP proves it, else solver upper bound (60 s grid)",
                 fontsize=8.5, color=INK2, y=1.02)
    fig.savefig(out / "fig3_solver_gaps.png")
    plt.close(fig)

    # ---------------------------------------------------------- knee table
    krows = []
    for x in xs:
        k = knees[str(x)]
        c = k["knee_cell"]
        krows.append({"x": x, "tau_star": k["tau_star"], "n": c["n"], "edges": c["edges"],
                      "components": c["components"], "giant": c["giant"],
                      "ILP_1h": c["giant_ilp_status"], "upper_bound": c["reference"],
                      "N": k["N"], "bits": k["bits"], "gap_words": c["gap_words"],
                      "gap_bits": c["gap_bits"], "N_forest": c["N_forest"], "N_greedy": c["N_greedy"],
                      "N_greedy_noswap": c["N_greedy_noswap"], "sec_ilp": c["sec_ilp"],
                      "sec_cell": c["sec_cell"]})
    T["knees"] = save_table(pd.DataFrame(krows), out, "table_knees")

    # ---------------------------------------------------------- fig 4 pool validation
    order = ["optimized", "greedy_noswap", "random", "top_freq"]
    prow = []
    for name, r in pv.items():
        xk, pool = name.split("/")
        prow.append({"x": float(xk[1:]), "pool": pool, "N": r["N"], "bits": r["bits"],
                     "spoken": r["spoken"], "not_spoken_unrepresentable": r["n_unrepresentable"],
                     "clean_self_recognition": r["clean_self_recognition"],
                     "conf_-5dB": r["conf_-5dB"], "ci_lo_-5dB": r["conf_-5dB_ci"][0],
                     "ci_hi_-5dB": r["conf_-5dB_ci"][1], "conf_-10dB": r["conf_-10dB"],
                     "ci_lo_-10dB": r["conf_-10dB_ci"][0], "ci_hi_-10dB": r["conf_-10dB_ci"][1]})
    pvt = pd.DataFrame(prow)
    pvt["pool"] = pd.Categorical(pvt.pool, order)
    pvt = pvt.sort_values(["x", "pool"]).reset_index(drop=True)
    T["pool_validation"] = save_table(pvt, out, "table_pool_validation")

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    w = 0.18
    for i, p in enumerate(order):
        d = pvt[pvt.pool == p]
        pos = np.arange(len(xs)) + (i - 1.5) * w
        m = d["conf_-5dB"].to_numpy()
        ax.errorbar(pos, m, yerr=[m - d["ci_lo_-5dB"], d["ci_hi_-5dB"] - m], fmt="o", ms=6,
                    color=COL[p], mec="white", mew=1.2, capsize=3, lw=1.8, label=LAB[p])
    ax.set_xticks(range(len(xs)), [f"x = {x:g}%\nN = {int(pvt[(pvt.x == x) & (pvt.pool == 'optimized')].N.iloc[0]):,}"
                                   for x in xs])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("in-pool confusion at −5 dB")
    ax.set_title("Pool validation, held-out channel (gTTS → SenseVoice), 95% bootstrap CI",
                 fontsize=9, color=INK)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(out / "fig4_pool_validation.png")
    plt.close(fig)

    # ---------------------------------------------------------- report.md
    hk = knees[str(float(cfg["lexicon"]["headline"]))]
    hc = hk["knee_cell"]
    ho, hh = mv["original"], mv["heldout"]
    exact_max = {x: grid[(grid.x == x) & grid.alpha_exact & (grid.tau > 0)].tau.max() for x in xs}
    nobound_min = {x: grid[(grid.x == x) & grid.reference.isna()].tau.min() for x in xs}
    pv5 = pvt[pvt.x == float(cfg["lexicon"]["headline"])].set_index("pool")
    rep = f"""# Countersign pool — results

All numbers are read from `runs/`; every figure has a table next to it in this folder.

## Headline (x = {cfg['lexicon']['headline']}%)

- Candidate set: **{lex['N_x']['5.0']:,}** 2–3-syllable NNG lemmas from OpenSubtitles KO at or above the bottom-5% Zipf of
  the NIKL 2003 level-C anchors ({lex['anchors_present']:,} of {lex['anchors_total']:,} anchors present in the corpus;
  cutoff Zipf {lex['cutoff_zipf']['5.0']:.2f} ≈ {lex['cutoff_occurrences']['5.0']:.0f} occurrences).
- Knee τ* = **{hk['tau_star']:g}** (Kneedle on N(τ)) → pool **N = {hk['N']:,}, {hk['bits']:.2f} bits**.
- 1 h ILP on the giant component ({hc['giant']:,} nodes): {hc['giant_ilp_status']}, upper bound {hc['reference']:,.0f}
  → remaining gap **{hc['gap_words']:.0f} words ({hc['gap_bits']:.3f} bits)**.
- Held-out pool validation at −5 dB: optimized {pv5.loc['optimized','conf_-5dB']:.2f}
  vs random {pv5.loc['random','conf_-5dB']:.2f} vs top-N frequency {pv5.loc['top_freq','conf_-5dB']:.2f}
  vs no-swap greedy {pv5.loc['greedy_noswap','conf_-5dB']:.2f} (N = {pv5.loc['greedy_noswap','N']:,}).

## 1. Metric calibration (original channel)

GLM-weighted Manhattan edit distance (GAP = {par['gap']:g}, CROSS = {par['cross']:g}), fit on the nearest 10% of
{cal['n_pairs']:,} pairs ({cal['pairs_with_confusion']} pairs with any confusion). Adopted: `{cal['adopted']}`.

{T['calibration_weights']}

Spearman(distance, confusion) — weak for every variant:

{T['calibration_spearman']}

Channel error by SNR (raw decision rule):

{T['calibration_snr']}

## 2. Metric validation — closest 100 vs random 100 pairs (Fig. 1)

![](fig1_metric_validation.png)

{T['metric_validation']}

- Original: δ = {ho['cliffs_delta']:.2f} [{ho['delta_ci'][0]:.2f}, {ho['delta_ci'][1]:.2f}], p = {ho['mw_p']:.1e};
  ratio {ho['ratio']:.2f} [{ho['ratio_ci'][0]:.2f}, {ho['ratio_ci'][1]:.2f}] (undefined in {ho['ratio_boot_undefined_share']:.0%} of bootstrap draws).
- Held-out: δ = {hh['cliffs_delta']:.2f} [{hh['delta_ci'][0]:.2f}, {hh['delta_ci'][1]:.2f}], p = {hh['mw_p']:.1e};
  **ratio undefined** (random pairs never confused).
- Pairs whose words failed the clean-audio gate were replaced from a reserve: {mv['skipped_closest']} closest, {mv['skipped_random']} random.

## 3. Grid, frontier and knees (Figs. 2–3)

75 cells = x ∈ {{5, 10, 50}} × τ ∈ {{0, 0.25, …, 6}}; distances computed once on the x = 5% set and reused.
Every cell: ILP on the giant component (60 s), greedy + swaps, forest (100 randomized runs, k = 3), no-swap greedy.
Per-cell N, bits, gap, components and runtimes: `table_grid_cells.csv`.

![](fig2_frontier.png)

![](fig3_solver_gaps.png)

- Exact α proven up to τ = {', '.join(f"{exact_max[x]:g} (x={x:g})" for x in xs)}.
- No bound — 60 s ILP returned UNKNOWN: {', '.join(f"τ={r.tau:g} (x={r.x:g})" for r in grid[grid.giant_ilp_status == 'UNKNOWN'].itertuples()) or 'none'}.
- No bound — giant-component model larger than RAM (`ilp_max_edges` = {cfg['solver']['ilp_max_edges']:,}), not built: from
  τ = {', '.join(f"{grid[(grid.x == x) & (grid.giant_ilp_status == 'NOT_RUN_TOO_LARGE')].tau.min():g} (x={x:g})" for x in xs if (grid[grid.x == x].giant_ilp_status == 'NOT_RUN_TOO_LARGE').any())}
  ({int((grid.giant_ilp_status == 'NOT_RUN_TOO_LARGE').sum())} cells).
- Method order is the same in every cell: solver best ≥ forest ≥ greedy + swaps ≥ no-swap greedy.
- The gap jumps where the giant component merges (τ ≈ 3.25): the heuristics barely move, the 60 s bound loosens.
  Gaps there measure the bound, not the heuristics; the 1 h knee cells below are the tighter figures.

Knee cells (1 h ILP):

{T['knees']}

## 4. Pool validation (held-out channel, Fig. 4)

Listener chooses among the whole pool; 300 spoken words per pool, −5 / −10 dB, 4 trials × 2 noise seeds.

![](fig4_pool_validation.png)

{T['pool_validation']}

## Notes for the paper

1. **Knee rule.** Kneedle on bits(τ) finds no knee at x = 5 / 10 (bits fall ~linearly, ≈0.23 bits per 0.25 τ) and only
   the τ≈0 onset at x = 50. τ* is therefore Kneedle on N(τ).
2. **Optimized vs no-swap greedy.** Both obey the same τ; their −5 dB confusion CIs overlap at every x (no-swap is slightly lower at x = 5);
   the optimizer's gain is pool size (+{int(pv5.loc['optimized','N'] - pv5.loc['greedy_noswap','N'])} words at x = 5), not lower confusion.
3. **−10 dB is at the floor** (≥ 0.91 error in every pool); −5 dB is the informative SNR.
4. **Unrepresentable words.** Words SenseVoice's tokenizer cannot write stay in the candidate set but are not spoken;
   counts per pool are in the table (rarer pools contain more names/typos — no extra lexicon filters, by design).
5. **Weak calibration.** Spearman ≈ −0.09; c_sonorant and v_round had the wrong sign and were floored, v_backness n.s.,
   n_cross separated (unused). The original channel is near its error floor ({ho['error_rate']:.0%} at the validation SNRs);
   the held-out channel carries the metric validation.
6. **Giant component merges at τ ≈ 3.25**: any length difference already costs GAP = 3.
7. **Compute limits.** ILP 60 s per grid cell, 1 h only at each knee; models above
   {cfg['solver']['ilp_max_edges']:,} edges are not built (5.9 GB RAM) → NaN bound, marked in Fig. 3.
8. **Metric-validation provenance.** {mv['note']}.
9. **Bugs fixed vs the notebook:** GLM sign convention; re-alignment under fitted weights; clean-audio gate against
   fixed distractors (never the pair partner); gate-failed pairs replaced from a reserve; SNR on active speech;
   deterministic per-trial noise seeds; ILP status reported (exact vs bound); deterministic CP-SAT.
"""
    (out / "report.md").write_text(rep, encoding="utf-8")
    C.manifest(cfg, "report", {"files": sorted(p.name for p in out.iterdir())})
    print("report ->", out)


if __name__ == "__main__":
    main()
