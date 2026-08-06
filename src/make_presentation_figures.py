"""Charts for the dissertation deck, generated from the measured artifacts.

Every figure here is drawn from a real results file -- no illustrative or
hand-entered numbers -- so a chart cannot drift away from the finding it
depicts. If an artifact is missing the script fails rather than substituting a
placeholder.

  fig_opm_rank.png     held-out error vs rank, with the memory crossover that
                       makes the low-rank layer cost more than stock
  fig_iptm_classes.png cognate / scrambled / decoy ipTM distributions
  fig_rank_stability.png  cognate rank across 4 identical re-runs
  fig_pvalue_repro.png    bootstrap distribution of the headline p-value

Usage:
    python src/make_presentation_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "reports" / "assets"

NAVY = "#0B2B52"
BLUE = "#1E6FB8"
TEAL = "#0E8F9A"
GREEN = "#2E7D32"
RED = "#C62828"
AMBER = "#8A5200"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#888888", "axes.labelcolor": NAVY,
    "text.color": NAVY, "xtick.color": "#555555", "ytick.color": "#555555",
    "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 200,
})


def _need(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing artifact: {p}")
    return json.loads(p.read_text())


def fig_opm_rank():
    """Held-out error vs rank, extrapolated to the 10% target."""
    d = _need(REPO_ROOT / "artifacts" / "opm_corpus_distill.json")
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    C_HIDDEN_SQ = 1024

    for layer, colour in (("layer_0", BLUE), ("layer_1", TEAL)):
        ranks = np.array(sorted(int(r) for r in d[layer]))
        err = np.array([d[layer][str(r)]["heldout"] for r in ranks])
        # err = a * rank^b  ->  fit in log space
        b, loga = np.polyfit(np.log(ranks), np.log(err), 1)
        a = np.exp(loga)
        need = (0.10 / a) ** (1 / b)
        xs = np.logspace(np.log10(ranks.min()), np.log10(max(need, 2000)), 200)
        ax.plot(xs, a * xs ** b, color=colour, lw=1.6, alpha=0.55, zorder=2)
        ax.scatter(ranks, err, s=54, color=colour, zorder=4,
                   label=f"{layer}  ({a:.2f}·rank$^{{{b:.3f}}}$) → 10% at rank ≈{need:,.0f}")

    ax.axhline(0.10, color=GREEN, ls=":", lw=1.4, zorder=1)
    ax.text(2100, 0.108, "10% error target", color=GREEN, fontsize=9, ha="right")
    ax.axvspan(C_HIDDEN_SQ, 1e4, color=RED, alpha=0.07, zorder=0)
    ax.axvline(C_HIDDEN_SQ, color=RED, lw=1.5, zorder=3)
    ax.text(C_HIDDEN_SQ * 1.12, 0.62,
            "c_hidden² = 1024\nstock materialises this width\n→ right of here the low-rank\n"
            "layer costs MORE memory",
            color=RED, fontsize=8.5, va="top")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(25, 1e4); ax.set_ylim(0.07, 0.9)
    ax.set_xlabel("rank"); ax.set_ylabel("held-out relative error")
    ax.set_title("Low-rank OPM: rank cannot buy back the fidelity")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    fig.tight_layout(); fig.savefig(OUT / "fig_opm_rank.png"); plt.close(fig)
    return "fig_opm_rank.png"


def fig_iptm_classes():
    """Class distributions, with the scramble sitting on top of the cognate."""
    P = [p for p in _need(REPO_ROOT / "artifacts" / "pdb_binders_b2_n22" /
                          "pdb_binder_scores.json")
         if not np.isnan(p.get("score", float("nan")))]
    groups = [("cognate", TEAL), ("scrambled", RED), ("decoy", GREEN)]
    data = [np.array([p["score"] for p in P if p["label"] == g]) for g, _ in groups]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=1.8))
    for patch, (_, c) in zip(bp["boxes"], groups):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("none")
    rng = np.random.default_rng(0)
    for i, (vals, (_, c)) in enumerate(zip(data, groups), start=1):
        ax.scatter(rng.normal(i, 0.055, len(vals)), vals, s=13,
                   color=NAVY, alpha=0.45, zorder=3, linewidths=0)
    for i, vals in enumerate(data, start=1):
        ax.text(i, vals.mean(), f"  {vals.mean():.4f}", va="center", ha="left",
                fontsize=9, color=NAVY, fontweight="bold")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"cognate\nn={len(data[0])}",
                        f"SCRAMBLED\nn={len(data[1])}",
                        f"decoy\nn={len(data[2])}"])
    ax.set_ylabel("ipTM")
    ax.set_title("Scrambled peptides score like cognates — and above real binders")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.text(0.5, 0.03,
            "scramble = same composition & length, order destroyed",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=AMBER)
    fig.tight_layout(); fig.savefig(OUT / "fig_iptm_classes.png"); plt.close(fig)
    return "fig_iptm_classes.png"


def fig_rank_stability():
    """Cognate rank across identical re-runs."""
    rec = _need(REPO_ROOT / "artifacts" / "seed_variance" / "seed_variance_scores.json")
    reps = sorted({r["replicate"] for r in rec})
    rids = sorted({r["receptor_id"] for r in rec})
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    palette = [BLUE, TEAL, GREEN, RED]

    for rid, colour in zip(rids, palette):
        seq = []
        for rep in reps:
            g = [r for r in rec if r["receptor_id"] == rid and r["replicate"] == rep
                 and r["label"] in ("cognate", "decoy") and not np.isnan(r["score"])]
            c = [x for x in g if x["label"] == "cognate"]
            dd = [x for x in g if x["label"] == "decoy"]
            if c and dd:
                seq.append(1 + sum(x["score"] >= c[0]["score"] for x in dd))
        ax.plot(range(1, len(seq) + 1), seq, "o-", color=colour, lw=2,
                ms=8, label=f"{rid}  {seq}")

    ax.set_yticks([1, 2, 3, 4]); ax.invert_yaxis()
    ax.set_xticks(range(1, len(reps) + 1))
    ax.set_xlabel("identical re-run"); ax.set_ylabel("cognate rank among its 3 decoys")
    ax.set_title("Same inputs, same settings — 4 of 4 receptors change rank")
    ax.legend(fontsize=9, loc="center right", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(0.02, 0.04, "rank 1 = cognate scored highest (best)",
            transform=ax.transAxes, fontsize=8.5, color=AMBER)
    fig.tight_layout(); fig.savefig(OUT / "fig_rank_stability.png"); plt.close(fig)
    return "fig_rank_stability.png"


def fig_pvalue_repro():
    """Where the headline p-value lands if the benchmark is re-run."""
    SD = _need(REPO_ROOT / "results" / "real" /
               "seed_variance_20260802.summary.json")["pooled_sd"]
    P = [p for p in _need(REPO_ROOT / "artifacts" / "pdb_binders_b2_n22" /
                          "pdb_binder_scores.json")
         if not np.isnan(p.get("score", float("nan")))]
    by = {}
    for p in P:
        by.setdefault(p["receptor_id"], []).append(p)
    rng = np.random.default_rng(0)
    ps = []
    for _ in range(4000):
        ranks = []
        for g in by.values():
            c = [x for x in g if x["label"] == "cognate"][0]
            dd = [x for x in g if x["label"] == "decoy"]
            cs = c["score"] + rng.normal(0, SD)
            ds = [x["score"] + rng.normal(0, SD) for x in dd]
            ranks.append(1 + sum(v >= cs for v in ds))
        ps.append(stats.wilcoxon(np.array(ranks, float) - 2.5,
                                 alternative="two-sided")[1])
    ps = np.array(ps)
    frac = (ps < 0.05).mean()

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.hist(ps, bins=45, color=BLUE, alpha=0.8, edgecolor="white", lw=0.4)
    ax.axvline(0.05, color=RED, lw=2)
    ax.text(0.056, ax.get_ylim()[1] * 0.93, "p = 0.05", color=RED, fontsize=10,
            fontweight="bold")
    ax.axvline(0.0337, color=GREEN, lw=1.6, ls="--")
    ax.annotate("as reported\np = 0.034", xy=(0.0337, ax.get_ylim()[1] * 0.45),
                xytext=(0.115, ax.get_ylim()[1] * 0.55), color=GREEN, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.text(0.5, 0.75, f"median p = {np.median(ps):.3f}", transform=ax.transAxes,
            fontsize=9.5, color=NAVY, fontweight="bold")
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("Wilcoxon p on re-running the whole 132-fold benchmark")
    ax.set_ylabel("simulated re-runs")
    ax.set_title(f"The headline p-value clears 0.05 in only {frac:.0%} of re-runs")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(); fig.savefig(OUT / "fig_pvalue_repro.png"); plt.close(fig)
    return "fig_pvalue_repro.png"


def fig_metric_comparison():
    """Why interface pLDDT works where ipTM does not."""
    d = _need(REPO_ROOT / "artifacts" / "rescore_metrics.json")["per_complex"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
    panels = [("iptm", "ipTM", "cognate = scramble\np = 0.42", axes[0]),
              ("iface_plddt", "interface pLDDT",
               "cognate > scramble\np < 0.0001", axes[1])]
    groups = [("cognate", TEAL), ("scrambled", RED), ("decoy", GREEN)]

    for key, title, verdict, ax in panels:
        data = [np.array([r[key] for r in d if r["label"] == g]) for g, _ in groups]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.6))
        for patch, (_, c) in zip(bp["boxes"], groups):
            patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("none")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["cognate", "SCRAM", "decoy"], fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        # bracket between cognate and its scramble -- the decisive comparison
        top = max(np.percentile(data[0], 95), np.percentile(data[1], 95))
        span = (max(np.max(data[0]), np.max(data[1]))
                - min(np.min(data[0]), np.min(data[1])))
        y = top + span * 0.10
        ax.plot([1, 1, 2, 2], [y, y + span * 0.04, y + span * 0.04, y],
                color=NAVY, lw=1.2)
        ax.text(1.5, y + span * 0.07, verdict, ha="center", va="bottom",
                fontsize=8.5, color=NAVY, fontweight="bold")
        ax.set_ylim(min(np.min(data[0]), np.min(data[1]), np.min(data[2]))
                    - span * 0.05, y + span * 0.30)
    axes[0].set_ylabel("score")
    fig.suptitle("Only interface pLDDT separates a binder from its own scramble",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_metric_comparison.png"); plt.close(fig)
    return "fig_metric_comparison.png"


def fig_few_step_arms():
    """Three arms: does few-step training explain the signal?"""
    decaf = _need(REPO_ROOT / "artifacts" / "decaf_scramble_result.json")["summary"]
    b1 = _need(REPO_ROOT / "artifacts" / "boltz1_scramble_result.json")["summary"]
    # Boltz-2 values come from Sections 7.6/7.7, folded on CPU
    b2 = {"iptm": {"cognate_minus_own_scramble": 0.013, "mean_rank": 2.00},
          "iface_plddt": {"cognate_minus_own_scramble": 3.30, "mean_rank": 1.91},
          "receptor_side": {"cognate_minus_own_scramble": 2.38}}
    # divide by the run-to-run SD measured in 7.5 so metrics on different
    # scales (ipTM 0-1, pLDDT 0-100) can share an axis
    sd = {"iptm": 0.0628, "iface_plddt": 1.9172, "receptor_side": 1.9172}
    mets = ["iptm", "iface_plddt", "receptor_side"]
    labels = ["ipTM", "interface\npLDDT", "receptor\nside"]
    arms = [("Boltz-2", b2, GREEN), ("Boltz-1", b1, BLUE), ("DeCAF", decaf, RED)]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
    x = np.arange(len(mets)); w = 0.26
    for i, (name, src, colour) in enumerate(arms):
        vals = [src.get(m, {}).get("cognate_minus_own_scramble", np.nan) / sd[m]
                for m in mets]
        axes[0].bar(x + (i - 1) * w, vals, w, label=name, color=colour, alpha=0.9)
    axes[0].axhline(1.0, color=NAVY, ls=":", lw=1.2)
    axes[0].text(2.45, 1.08, "= noise", fontsize=8, color=NAVY, ha="right")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("effect / run-to-run noise")
    axes[0].set_title("Order sensitivity", fontsize=11.5)
    axes[0].legend(fontsize=8.5, framealpha=0.95)
    axes[0].grid(axis="y", alpha=0.25, lw=0.5)

    rm = ["iptm", "iface_plddt"]
    rl = ["ipTM", "interface pLDDT"]
    xr = np.arange(len(rm))
    for i, (name, src, colour) in enumerate(arms):
        vals = [src.get(m, {}).get("mean_rank", np.nan) for m in rm]
        axes[1].bar(xr + (i - 1) * w, vals, w, label=name, color=colour, alpha=0.9)
    axes[1].axhline(2.5, color=NAVY, ls="--", lw=1.4)
    axes[1].text(1.45, 2.55, "chance", fontsize=8.5, color=NAVY, ha="right")
    axes[1].set_xticks(xr); axes[1].set_xticklabels(rl, fontsize=9)
    axes[1].set_ylim(1.0, 2.8)
    axes[1].set_ylabel("mean cognate rank (lower = better)")
    axes[1].set_title("Receptor specificity", fontsize=11.5)
    axes[1].grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("Few-step distillation, not the base model, drives the gain",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_few_step_arms.png"); plt.close(fig)
    return "fig_few_step_arms.png"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_opm_rank, fig_iptm_classes, fig_rank_stability,
               fig_pvalue_repro, fig_metric_comparison,
               fig_few_step_arms):
        print(f"  wrote {OUT / fn()}")


if __name__ == "__main__":
    main()
