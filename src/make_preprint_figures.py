"""Figures 2 and 3 of the preprint, generated from the measured artifacts.

Same discipline as make_presentation_figures.py: every value is read from a
results file, and a missing artifact is a failure rather than a placeholder.

  fig_contamination.png   Cohen's d on the scramble control across the 2x2 of
                          {in-training, held out} x {reduced, full settings}
  fig_wetlab.png          (a) macro-AP against measured binding for every
                          predictor in the release, (b) the same readout family
                          across three regimes, on one AUC convention

One convention note, because it decides what panel (b) is allowed to plot.
Two within-group AUCs appear in this project:

  single_auc      z-score within receptor, then one global ROC, with decoys AND
                  scrambles as negatives. Sections 7.13 and 7.18 use this, and
                  it is what the wet-lab comparison uses.
  within_auc      P(cognate outranks a decoy of its own receptor), ties at half,
                  scrambles excluded. heldout_at_full.json stores this one.

Section 7.18.4's table takes its in-training row from the first and its held-out
row from the second, so its two rows are not on the same scale. Panel (b)
recomputes every regime with single_auc from per-fold scores instead.

Usage:
    python src/make_preprint_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "reports" / "assets"
ART = REPO_ROOT / "artifacts"
sys.path.insert(0, str(REPO_ROOT / "src"))
from model_ensemble import single_auc  # noqa: E402

NAVY = "#0B2B52"
BLUE = "#1E6FB8"
TEAL = "#0E8F9A"
GREEN = "#2E7D32"
RED = "#C62828"
AMBER = "#8A5200"
GREY = "#9AA5B1"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#888888", "axes.labelcolor": NAVY,
    "text.color": NAVY, "xtick.color": "#555555", "ytick.color": "#555555",
    "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 200,
})

READOUTS = [("iptm", "ipTM"),
            ("iface_plddt", "interface\npLDDT"),
            ("receptor_side", "receptor\nside")]


def _need(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing artifact: {p}")
    return json.loads(p.read_text())


def fig_contamination():
    """The contamination penalty, and the settings effect, in one 2x2."""
    d = _need(ART / "heldout_at_full.json")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), sharey=True)
    x = np.arange(len(READOUTS))
    w = 0.36

    for ax, (arm_in, arm_held, title) in zip(axes, [
            ("in_reduced", "held_reduced", "reduced: 10 sampling steps"),
            ("in_full", "held_full", "full: 200 sampling steps")]):
        din = [d[k][arm_in]["d"] for k, _ in READOUTS]
        dhe = [d[k][arm_held]["d"] for k, _ in READOUTS]
        ax.bar(x - w / 2, din, w, label="in training", color=BLUE)
        ax.bar(x + w / 2, dhe, w, label="held out", color=AMBER)
        # Cohen's conventional thresholds, so the bars can be read without
        # knowing the scale by heart.
        for y, lab in ((0.2, "small"), (0.5, "medium"), (0.8, "large")):
            ax.axhline(y, color=GREY, lw=0.7, ls=":", zorder=0)
            ax.text(len(READOUTS) - 0.42, y + 0.015, lab, fontsize=7,
                    color=GREY, ha="right", va="bottom")
        for xi, (a, b) in enumerate(zip(din, dhe)):
            ax.text(xi, max(a, b) + 0.07, f"{b / a:.0%}\nretained",
                    ha="center", va="bottom", fontsize=8, color=NAVY,
                    fontweight="bold", linespacing=1.15)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in READOUTS], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Cohen's $d$, cognate vs own scramble")
    axes[0].set_ylim(0, 2.05)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Sampling budget sets the effect size; training exposure sets "
                 "how much survives",
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_contamination.png")
    plt.close(fig)
    return "fig_contamination.png"


def _auc_panel(path, key):
    return single_auc(pd.DataFrame(_need(path)["per_fold"]), key)[1]


def fig_wetlab():
    """Against measured binding, and the descent that leads to it."""
    v = _need(ART / "anthropic_validation.json")
    base = v["base_rate"]

    fig = plt.figure(figsize=(12.4, 4.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.38, 1.0], wspace=0.34,
                      left=0.185, right=0.985)

    # (a) every predictor in the release, macro-AP with 95% CI
    ax = fig.add_subplot(gs[0, 0])
    preds = sorted(v["predictors"], key=lambda p: p["macro_ap"])
    labels = [p["label"].replace("  (this work's readout)", " *")
                        .replace("  (this work's model)", " *")
              for p in preds]
    aps = np.array([p["macro_ap"] for p in preds])
    lo = aps - np.array([p["ap_ci"][0] for p in preds])
    hi = np.array([p["ap_ci"][1] for p in preds]) - aps
    mine = [i for i, p in enumerate(preds) if "this work" in p["label"]]
    colours = [RED if i in mine else BLUE for i in range(len(preds))]
    y = np.arange(len(preds))
    ax.barh(y, aps, color=colours, height=0.62)
    ax.errorbar(aps, y, xerr=[lo, hi], fmt="none", ecolor="#33475B",
                elinewidth=1.0, capsize=2.5)
    ax.axvline(base, color=NAVY, lw=1.6, ls="--")
    ax.text(base + 0.008, -0.62, f"chance {base:.3f}", fontsize=8,
            color=NAVY, fontweight="bold", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("macro-AP against measured binding")
    ax.set_xlim(0, 0.72)
    ax.set_title(f"(a) {v['n_designs']:,} designs, {v['n_targets']} targets",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # (b) one readout family, three regimes, one AUC convention
    ax2 = fig.add_subplot(gs[0, 1])
    held = [_auc_panel(ART / f"heldout_panel_result_full{s}.json", k)
            for s in ("", "2") for k in ("iface_plddt",)]
    held_ip = [_auc_panel(ART / f"heldout_panel_result_full{s}.json", "iptm")
               for s in ("", "2")]
    tracks = [
        ("interface pLDDT", TEAL,
         [_auc_panel(ART / "settings_confound.json", "iface_plddt"),
          float(np.mean(held)),
          next(p["within_auc"] for p in v["predictors"]
               if p["label"].startswith("Interface pLDDT"))]),
        ("ipTM / ipSAE", BLUE,
         [_auc_panel(ART / "settings_confound.json", "iptm"),
          float(np.mean(held_ip)),
          next(p["within_auc"] for p in v["predictors"]
               if p["label"].startswith("Boltz-2"))]),
    ]
    xs = np.arange(3)
    # The tracks cross, so a fixed above/below rule collides at the crossing.
    # Each label goes on the outside of whichever track is higher at that x.
    other = {0: 1, 1: 0}
    for ti, (name, colour, ys) in enumerate(tracks):
        ax2.plot(xs, ys, "-o", color=colour, lw=2.0, ms=7, label=name)
        rival = tracks[other[ti]][2]
        for xi, yi in zip(xs, ys):
            dy = 11 if yi >= rival[xi] else -19
            ax2.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                         xytext=(0, dy), ha="center", fontsize=8.5,
                         color=colour, fontweight="bold")
    ax2.axhline(0.5, color=GREY, lw=1.2, ls="--")
    ax2.text(2.32, 0.512, "chance", fontsize=8, color=GREY, ha="right")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["in training\n(full settings)",
                         "held out\n(full settings)",
                         "measured\nbinding"], fontsize=9)
    ax2.set_xlim(-0.28, 2.38)
    ax2.set_ylim(0.40, 1.0)
    ax2.set_ylabel("within-group AUC")
    ax2.set_title("(b) the same readout, three regimes", fontsize=11)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.grid(axis="y", alpha=0.25, lw=0.5)
    ax2.set_axisbelow(True)

    fig.suptitle("No confidence readout is far above chance on binding that was "
                 "actually measured",
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.savefig(OUT / "fig_wetlab.png", bbox_inches="tight")
    plt.close(fig)
    return "fig_wetlab.png"


if __name__ == "__main__":
    for fn in (fig_contamination, fig_wetlab):
        print(f"  wrote {fn()}")
