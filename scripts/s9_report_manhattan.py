"""Step 9 — report for the adopted pipeline (Manhattan distance on syllable-slot vectors).

Reads only stage outputs (no recomputation):
  runs_manhattan/calibration_v2/       GLM calibration of the edit distance (refit)
  runs/metric_selection/               4 metrics, original channel (selection)
  runs_manhattan/metric_selection_heldout/   same design, held-out channel (confirmation)
  runs_manhattan/grid, pools, pool_validation
  runs/grid, runs/pool_validation      superseded edit-distance pipeline (appendix)
Writes runs_manhattan/report/: fig1..fig4 (.png), table_*.csv/.md, report.md

Run with CS_CONFIG=config_manhattan.yaml.
"""
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

COL = {"optimized": "#2a78d6", "random": "#eb6834", "top_freq": "#1baf7a",
       "greedy_noswap": "#eda100", "greedy": "#e87ba4", "forest": "#008300"}
LAB = {"optimized": "optimized (solver best)", "random": "random", "top_freq": "top-N frequency",
       "greedy_noswap": "greedy, no swaps", "greedy": "greedy + swaps", "forest": "forest (100 runs)"}
MCOL = {"calibrated_edit": "#2a78d6", "manhattan": "#eb6834", "euclidean": "#1baf7a", "cosine": "#eda100"}
MLAB = {"calibrated_edit": "calibrated edit", "manhattan": "Manhattan (vector)",
        "euclidean": "Euclidean (vector)", "cosine": "cosine (vector)"}
XCOL = {5.0: "#104281", 10.0: "#2a78d6", 50.0: "#86b6ef"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def md_table(df, floatfmt=3):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "—" if math.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_table(df, out, name, floatfmt=3):
    df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    (out / f"{name}.md").write_text(md_table(df, floatfmt), encoding="utf-8")
    return md_table(df, floatfmt)


def ci(v):
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


def main():
    cfg = C.load()
    R = Path(cfg["out_dir"])                       # runs_manhattan
    OLD = Path("runs")
    out = C.stage(cfg, "report")
    lex = C.load_json(R / "lexicon" / "summary.json")
    cal = C.load_json(R / "calibration_v2" / "result.json")
    par = C.load_json(R / "calibration_v2" / "params.json")
    sel = {"original": C.load_json(OLD / "metric_selection" / "result.json"),
           "heldout": C.load_json(R / "metric_selection_heldout" / "result.json")}
    grid = pd.read_csv(R / "grid" / "grid.csv")
    knees = C.load_json(R / "grid" / "knee.json")
    pv = C.load_json(R / "pool_validation" / "result.json")
    xs = sorted(grid.x.unique())
    step = float(cfg["grid"]["tau"][2])
    metrics = cfg["metric_selection"]["metrics"]
    H = float(cfg["lexicon"]["headline"])
    T = {}

    # ------------------------------------------------------------ 1 calibration
    last = cal["history"][-1]
    names = ["c_place", "c_manner", "c_laryngeal", "c_sonorant", "v_height", "v_backness", "v_round", "v_glide"]
    cal_tab = pd.DataFrame({"feature": names,
                            "glm_coef": [last["coef"][n] for n in names],
                            "sign": [last["signs"][n] for n in names],
                            "weight_adopted": list(par["cons_w"]) + list(par["vowel_w"])})
    T["cal_w"] = save_table(cal_tab, out, "table_calibration_weights")
    T["cal_sp"] = save_table(pd.DataFrame({"weights": list(cal["spearman"]),
                                           "spearman_d_vs_confusion": list(cal["spearman"].values())}),
                             out, "table_calibration_spearman")
    T["cal_snr"] = save_table(pd.DataFrame({"snr_db": [int(k) for k in cal["error_by_snr_raw"]],
                                            "error_rate": list(cal["error_by_snr_raw"].values())})
                              .sort_values("snr_db", ascending=False), out, "table_calibration_error_by_snr")

    # ------------------------------------------------------------ 2 metric selection
    rows = []
    for ch, r in sel.items():
        for m in metrics:
            q = r["metrics"][m]
            rows.append({"channel": ch, "metric": m, "rho_all": q["spearman_all"],
                         "rho_all_lo": q["spearman_all_ci"][0], "rho_all_hi": q["spearman_all_ci"][1],
                         "rho_near": q["spearman_near"],
                         "rho_near_lo": q["spearman_near_ci"][0], "rho_near_hi": q["spearman_near_ci"][1],
                         "minus_winner_lo": q["minus_winner_ci"][0], "minus_winner_hi": q["minus_winner_ci"][1],
                         "rho_0dB": q["spearman_all_by_snr"]["0"], "rho_-5dB": q["spearman_all_by_snr"]["-5"],
                         "channel_winner": r["winner"]})
    selt = pd.DataFrame(rows)
    T["sel"] = save_table(selt.drop(columns="channel_winner"), out, "table_metric_selection")
    sinfo = pd.DataFrame([{"channel": ch, "pairs": sum(r["n_pairs"].values()), "words": r["n_eval_words"],
                           "pairs_with_confusion": r["pairs_with_confusion"], "error_rate": r["error_rate"],
                           "error_0dB": r["error_by_snr"]["0"], "error_-5dB": r["error_by_snr"]["-5"],
                           "gate_replacements": sum(r["skipped"].values()), "winner": r["winner"]}
                          for ch, r in sel.items()])
    T["sel_info"] = save_table(sinfo, out, "table_metric_selection_design")
    nr = sel["original"]["near_ranges"]
    T["near"] = save_table(pd.DataFrame([{"metric": m, "near_range_pairs": nr[m]["size"],
                                          "d_min": nr[m]["d_min"], "d_max": nr[m]["d_max"]} for m in metrics]),
                           out, "table_metric_near_ranges")

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    for ax, (ch, title) in zip(axs, [("original", "Original: edge-tts → Whisper-small\n(selection)"),
                                     ("heldout", "Held-out: gTTS → SenseVoice-Small\n(confirmation)")]):
        d = selt[selt.channel == ch].set_index("metric")
        for k, m in enumerate(metrics):
            y = len(metrics) - 1 - k
            r = d.loc[m]
            ax.errorbar(r.rho_all, y + 0.12, xerr=[[r.rho_all - r.rho_all_lo], [r.rho_all_hi - r.rho_all]],
                        fmt="o", ms=6.5, color=MCOL[m], mec="white", mew=1.2, capsize=3, lw=1.8)
            ax.errorbar(r.rho_near, y - 0.12, xerr=[[r.rho_near - r.rho_near_lo], [r.rho_near_hi - r.rho_near]],
                        fmt="o", ms=5.5, color=MCOL[m], mfc="white", mew=1.4, capsize=3, lw=1.2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("Spearman ρ (distance, pair confusion)")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(metrics)), [MLAB[m] for m in metrics[::-1]])
    axs[1].plot([], [], "o", color=INK2, label="all pairs (primary)")
    axs[1].plot([], [], "o", color=INK2, mfc="white", label="near pairs only")
    axs[1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Metric selection: more negative = distance predicts confusion better; 95% bootstrap CI",
                 fontsize=8.5, color=INK2, y=1.04)
    fig.savefig(out / "fig1_metric_selection.png")
    plt.close(fig)

    # ------------------------------------------------------------ grid table
    cell = grid.assign(
        ILP=grid.giant_ilp_status.fillna("none_needed"),
        reference_kind=np.where(grid.alpha_exact, "exact α",
                                np.where(grid.reference.isna(), "no bound", "upper bound")),
    )[["x", "tau", "n", "edges", "components", "giant", "ILP", "reference_kind", "reference",
       "N", "bits", "gap_words", "gap_bits", "N_greedy", "N_forest", "N_greedy_noswap",
       "sec_greedy", "sec_forest", "sec_ilp", "sec_cell"]]
    save_table(cell, out, "table_grid_cells")

    # ------------------------------------------------------------ fig 2 frontier
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for x in xs:
        g = grid[grid.x == x].sort_values("tau")
        k = knees[str(x)]
        ax.plot(g.tau, g.bits, "-", color=XCOL[x], lw=2,
                label=f"x = {x:g}%  (n = {int(g.n.iloc[0]):,}; knee τ*={k['tau_star']:g}, "
                      f"N={k['N']:,} = {k['bits']:.2f} bits)")
        ax.plot(g.tau, g.bits, "o", color=XCOL[x], ms=3.5)
        ax.plot(k["tau_star"], k["bits"], "o", ms=10, color=XCOL[x], mec="white", mew=2, zorder=6)
    ax.set_xlabel("threshold τ (edge iff Manhattan distance < τ)")
    ax.set_ylabel("pool entropy, bits = log₂ N")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Frontier: pool size vs τ (knee: Kneedle on N(τ), convex part)", fontsize=9, color=INK)
    fig.savefig(out / "fig2_frontier.png")
    plt.close(fig)

    # ------------------------------------------------------------ fig 3 solver gaps
    fig, axs = plt.subplots(1, len(xs), figsize=(9, 2.9), sharey=True)
    for ax, x in zip(axs, xs):
        g = grid[grid.x == x].sort_values("tau")
        ref = g.reference.to_numpy(float)
        for m in ("optimized", "forest", "greedy", "greedy_noswap"):
            Nm = g.N.to_numpy(float) if m == "optimized" else g[f"N_{m}"].to_numpy(float)
            ax.plot(g.tau, np.log2(ref) - np.log2(Nm), "-o", ms=3, lw=1.8, color=COL[m], label=LAB[m])
        nob = g.tau[g.reference.isna()]
        if len(nob):
            ax.axvspan(nob.min() - step / 2, g.tau.max() + step / 2, color=GRID, alpha=0.6, lw=0)
        ax.axvline(knees[str(x)]["tau_star"], color=INK2, lw=0.8, ls="--")
        ax.set_title(f"x = {x:g}%  (dashed: τ*)", fontsize=9, color=INK)
        ax.set_xlabel("τ")
    for ax, x in zip(axs, xs):
        nob = grid[(grid.x == x) & grid.reference.isna()].tau
        if len(nob):
            ax.text(nob.min() - step / 2, ax.get_ylim()[1], " no bound", va="top", fontsize=7, color=INK2)
    axs[0].set_ylabel("gap to reference (bits)")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Reference = exact α where the ILP proves it, else the solver's upper bound (60 s)",
                 fontsize=8.5, color=INK2, y=1.02)
    fig.savefig(out / "fig3_solver_gaps.png")
    plt.close(fig)

    krows = []
    for x in xs:
        k = knees[str(x)]
        c = k["knee_cell"]
        krows.append({"x": x, "tau_star": k["tau_star"], "fit": k["curve"], "n": c["n"], "edges": c["edges"],
                      "components": c["components"], "giant": c["giant"], "ILP": c["giant_ilp_status"],
                      "reference": c["reference"], "N": k["N"], "bits": k["bits"],
                      "gap_words": c["gap_words"], "gap_bits": c["gap_bits"], "N_forest": c["N_forest"],
                      "N_greedy": c["N_greedy"], "N_greedy_noswap": c["N_greedy_noswap"],
                      "sec_ilp": c["sec_ilp"], "sec_cell": c["sec_cell"]})
    knt = pd.DataFrame(krows)
    T["knees"] = save_table(knt, out, "table_knees")

    # ------------------------------------------------------------ fig 4 pool validation
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
    T["pv"] = save_table(pvt, out, "table_pool_validation")

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    w = 0.18
    for i, p in enumerate(order):
        d = pvt[pvt.pool == p]
        pos = np.arange(len(xs)) + (i - 1.5) * w
        m = d["conf_-5dB"].to_numpy()
        ax.errorbar(pos, m, yerr=[m - d["ci_lo_-5dB"], d["ci_hi_-5dB"] - m], fmt="o", ms=6,
                    color=COL[p], mec="white", mew=1.2, capsize=3, lw=1.8, label=LAB[p])
    ax.set_xticks(range(len(xs)), [f"x = {x:g}%, τ* = {knees[str(x)]['tau_star']:g}\n"
                                   f"N = {knees[str(x)]['N']:,}" for x in xs])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("in-pool confusion at −5 dB")
    ax.set_title("Pool validation, held-out channel (gTTS → SenseVoice), 95% bootstrap CI",
                 fontsize=9, color=INK)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(out / "fig4_pool_validation.png")
    plt.close(fig)

    # ------------------------------------------------------------ appendix: superseded pipeline
    og = pd.read_csv(OLD / "grid" / "grid.csv")
    ok = C.load_json(OLD / "grid" / "knee.json")
    opv = C.load_json(OLD / "pool_validation" / "result.json")
    arows = []
    for x in sorted(og.x.unique()):
        k = ok[str(x)]
        row = {"x": x, "tau_star": k["tau_star"], "N": k["N"], "bits": k["bits"],
               "gap_words": k["knee_cell"]["gap_words"]}
        for p in order:
            q = opv.get(f"x{x:g}/{p}")
            row[f"conf_-5dB_{p}"] = q["conf_-5dB"] if q else float("nan")
        arows.append(row)
    T["old"] = save_table(pd.DataFrame(arows), out, "table_appendix_edit_pipeline")

    # ------------------------------------------------------------ report.md
    hk = knees[str(H)]
    hc = hk["knee_cell"]
    so, sh = sel["original"]["metrics"], sel["heldout"]["metrics"]
    exact_max = {x: grid[(grid.x == x) & grid.alpha_exact & (grid.tau > 0)].tau.max() for x in xs}
    pvh = pvt[pvt.x == H].set_index("pool")
    unk = grid[grid.giant_ilp_status == "UNKNOWN"]
    big = grid[grid.giant_ilp_status == "NOT_RUN_TOO_LARGE"]
    old5 = ok[str(H)]
    rep = f"""# Countersign pool — results (adopted pipeline: Manhattan distance)

Every number is read from the stage outputs; each figure has its table next to it in this folder.

## Headline (x = {H:g}%)

- Candidate set: **{lex['N_x']['5.0']:,}** 2–3-syllable NNG lemmas (OpenSubtitles KO, Kiwi) at or above the bottom-5% Zipf
  of the NIKL anchors (cutoff Zipf {lex['cutoff_zipf']['5.0']:.2f}).
- Distance: **Manhattan (L1) on fixed-length articulatory vectors** — chosen over the GLM-calibrated edit distance,
  Euclidean and cosine in §2.
- Knee τ* = **{hk['tau_star']:g}** (edges = pairs with distance ≤ {hk['tau_star'] - 1:g}) → pool **N = {hk['N']:,},
  {hk['bits']:.2f} bits**; solver upper bound {hc['reference']:,.0f} → gap ≤ **{hc['gap_words']:.0f} words
  ({hc['gap_bits']:.3f} bits)**.
- Held-out pool validation, −5 dB: optimized **{pvh.loc['optimized','conf_-5dB']:.2f}**
  {ci([pvh.loc['optimized','ci_lo_-5dB'], pvh.loc['optimized','ci_hi_-5dB']])} vs random
  {pvh.loc['random','conf_-5dB']:.2f} vs top-N frequency {pvh.loc['top_freq','conf_-5dB']:.2f};
  no-swap greedy {pvh.loc['greedy_noswap','conf_-5dB']:.2f} (N = {pvh.loc['greedy_noswap','N']:,}).

## 1. Calibrating the edit distance (GLM)

The candidate "calibrated edit" distance is a weighted Manhattan edit distance between surface forms: substitution
cost = weighted L1 difference of the 4 articulatory features (consonant: place, manner, laryngeal, sonorant; vowel:
height, backness, round, glide), gap = {par['gap']:g}, consonant↔vowel crossing = {par['cross']:g}.

**How the weights are fit.** {cal['n_pairs']:,} word pairs from 320 sampled words were played through the original
channel (edge-tts → Whisper-small, forced choice, 0 / −5 dB, 8 trials); per pair the outcome is the number of
confusions out of the number of presentations. Each pair is aligned under the current weights and summarised as
*per-feature mismatch totals along the alignment* plus counts of gaps and crossings. A **binomial GLM (logit link)**
regresses confusion on those totals, fit on the nearest {cal['fit_scope']} pairs (far pairs are never confused and
carry no information about weights). A negative coefficient means "a mismatch on this feature makes confusion less
likely", i.e. the feature separates words, so **weight = max(−coef, 0.1)**; a wrong-sign (positive) coefficient is
floored, not flipped. Pairs are then re-aligned under the new weights and the GLM refit until the weights stop
changing ({len(cal['history'])} iterations). Three candidates — uniform, raw and mean-normalised weights — are
compared by Spearman ρ(distance, confusion) on all pairs; the best is adopted (`{cal['adopted']}`).

{T['cal_w']}

{T['cal_sp']}

Channel error by SNR:

{T['cal_snr']}

**Reading.** All coefficients but v_glide are negative and significant (v_glide floored), but calibration improves ρ
only from {cal['spearman']['uniform']:.3f} (uniform) to {cal['spearman']['calibrated_norm']:.3f}. The weights
matter little; §2 asks whether the edit distance is needed at all.

## 2. Choosing the distance (Fig. 1)

Four distances, all on the same g2pk2 surface forms and the same articulatory features:

| metric | definition |
|---|---|
| calibrated edit | §1, with alignment (handles length differences by gaps) |
| Manhattan | L1 between 45-dim vectors: 3 syllables × (onset, vowel, coda) × (4 features + present bit), left-aligned, zero-padded |
| Euclidean | L2 between the same vectors |
| cosine | 1 − cosine similarity of the same vectors |

**Design (one shared pair set).** Each metric's *near range* = its 20,000 closest pairs (ties at the boundary kept).
50 pairs are drawn at random from each metric's near range plus 50 random pairs → 250 pairs; words used to fit §1
and identical-surface pairs (distance 0 under every metric, {sel['original']['identical_surface_pairs_excluded']})
are excluded. Every word passes a clean-audio gate (forced choice against 50 fixed distractors; a failed word's
pair is replaced from a reserve). All evaluated words form **one** candidate set; each is spoken 8× at 0 and −5 dB.
Pair confusion rate = (a heard as b + b heard as a) / (presentations of a + b). Every metric is scored on all 250
pairs, so the comparison is paired.

**Score and significance test.** Spearman ρ(distance, pair confusion rate) over all pairs (primary) and over the
200 near pairs (secondary). Uncertainty: **bootstrap over pairs** (2,000 resamples of the 250 pairs with
replacement; percentile 95% CI). Because all metrics are scored on the same resample, the CI of
**ρ(metric) − ρ(winner)** is a paired test: if it contains 0 the metric is not distinguishable from the winner at
the 5% level. Selection used the **original channel only**; the held-out channel was run afterwards with the same
design as a confirmation and did not influence the choice.

{T['sel_info']}

![](fig1_metric_selection.png)

{T['sel']}

- Original: Manhattan ρ = {so['manhattan']['spearman_all']:.3f}, calibrated edit {so['calibrated_edit']['spearman_all']:.3f};
  difference CI {ci(so['calibrated_edit']['minus_winner_ci'])} → **tied**. Euclidean and cosine are significantly worse
  ({ci(so['euclidean']['minus_winner_ci'])}, {ci(so['cosine']['minus_winner_ci'])}).
- Held-out: calibrated edit ρ = {sh['calibrated_edit']['spearman_all']:.3f}, Manhattan {sh['manhattan']['spearman_all']:.3f};
  difference CI {ci(sh['manhattan']['minus_winner_ci'])} → **tied, order reversed**. Euclidean and cosine again
  significantly worse.
- **Decision:** Manhattan and calibrated edit are statistically indistinguishable on both channels; Manhattan is
  adopted because it won on the selection channel and is simpler (no listening data, no fitted weights, no
  alignment). Euclidean and cosine are rejected on both channels.
- Random pairs were never confused on either channel; all signal comes from near pairs.

## 3. Grid, frontier and knees (Figs. 2–3)

{len(grid)} cells = x ∈ {{{', '.join(f'{x:g}' for x in xs)}}} × τ ∈ {{0, 1, …, 13}}; edge iff Manhattan distance < τ
(distances are integers, so τ = 7 means distance ≤ 6). Per cell: ILP (CP-SAT, 60 s) on the giant component, greedy +
swaps, forest (100 randomised runs), no-swap greedy; ties broken by frequency. Per-cell N, bits, gap, components and
runtimes: `table_grid_cells.csv`.

![](fig2_frontier.png)

![](fig3_solver_gaps.png)

- **Knee rule.** N(τ) is S-shaped (flat at τ ≤ 1, where only identical vectors connect; steepest drop; convex tail).
  Kneedle (convex, decreasing) is applied from the steepest drop onward. bits(τ) has no knee (≈ linear).
- Exact α proven up to τ = {', '.join(f"{exact_max[x]:g} (x={x:g})" for x in xs)}; beyond that the 60 s ILP gives an
  upper bound only, and the gap shown is mostly looseness of the bound.
- No bound: ILP returned UNKNOWN at {', '.join(f"τ={r.tau:g} (x={r.x:g})" for r in unk.itertuples()) or 'none'};
  model not built (giant component > {cfg['solver']['ilp_max_edges']:,} edges, 5.9 GB RAM) at
  {', '.join(f"τ={r.tau:g} (x={r.x:g})" for r in big.itertuples()) or 'none'}.
- Method order: solver best ≥ forest ≥ greedy + swaps ≥ no-swap greedy in every cell; all within ~1% of each other
  at the knees.

{T['knees']}

## 4. Pool validation (held-out channel, Fig. 4)

The listener (gTTS → SenseVoice-Small) chooses among the **whole pool**; 300 spoken words per pool, −5 / −10 dB.
Baselines of equal N: random words from the same candidate set, and the N most frequent words.

![](fig4_pool_validation.png)

{T['pv']}

- x = 5 and 10: optimized < random < top-N frequency; optimized vs random CIs just touch (modest effect).
- x = 50: optimized, random and top-N overlap — no measurable difference at N = {knees['50.0']['N']}.
- No-swap greedy matches optimized at every x: the solver's gain is pool size, not lower confusion.
- −10 dB is at the floor (≥ 0.96) in every pool; −5 dB is the informative SNR.
- Error is much higher than in the superseded pipeline (appendix) because the pools are larger (more alternatives
  per forced choice), not because the pools are worse.

## Notes for the paper

1. **Distance choice is a tie.** Report Manhattan as "not worse than the calibrated edit distance on either channel,
   and simpler", not as "best".
2. **Weak calibration.** Learned weights add little over uniform (ρ {cal['spearman']['uniform']:.3f} →
   {cal['spearman']['calibrated_norm']:.3f}); only {cal['pairs_with_confusion']} of {cal['n_pairs']:,} pairs were ever confused.
3. **Unrepresentable words.** SenseVoice's tokenizer cannot write some words; they stay in the pool but are not spoken.
   At x = 5 this is {int(pvh.loc['optimized','not_spoken_unrepresentable'])} words in the optimized pool,
   {int(pvh.loc['greedy_noswap','not_spoken_unrepresentable'])} no-swap, {int(pvh.loc['random','not_spoken_unrepresentable'])} random,
   {int(pvh.loc['top_freq','not_spoken_unrepresentable'])} top-N — the optimizer favours rare/odd forms. This affects
   which words are spoken and must be disclosed; at x = 50 it is ≤ 2 words.
4. **Bounds at the knee.** τ* lies beyond the range where the 60 s ILP proves α; the knee gaps are upper bounds
   ({hc['gap_bits']:.3f} bits at x = {H:g}).
5. **Bugs fixed vs the notebook:** GLM sign convention; re-alignment under fitted weights; clean-audio gate against
   fixed distractors; gate-failed pairs replaced; SNR on active speech; deterministic noise seeds; ILP status reported
   (exact vs bound); knee-shape detection (the first version labelled S-shaped curves concave and returned the grid
   edge — fixed before pool validation).

## Appendix — superseded edit-distance pipeline

Same steps with the earlier calibrated edit distance (τ grid 0–6 in steps of 0.25, 1 h ILP at knees).
At x = 50 its knee was chosen with the earlier, buggy shape rule (τ* = {ok['50.0']['tau_star']:g}; the fixed rule gives
2.75); x = 5 and 10 are unchanged. Not re-run.

{T['old']}
"""
    (out / "report.md").write_text(rep, encoding="utf-8")
    C.manifest(cfg, "report", {"files": sorted(p.name for p in out.iterdir())})
    print("report ->", out)


if __name__ == "__main__":
    main()
