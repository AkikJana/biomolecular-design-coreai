"""A release date after the cutoff does not make a structure novel to the model.

Section 7.10 builds a held-out panel by taking PDB entries released after
2021-09-30. That is a *temporal* split, and temporal splits alone are no longer
the standard: contamination-aware protein benchmarks pair them with sequence
decontamination, because a domain deposited in 2024 whose 95%-identical relative
was deposited in 2019 is not novel to a model trained on the 2019 entry.
ProteinArena admits only sequences under 30% identity to anything released
before its cutoff; other work uses MMseqs2 at 50% identity over 80% coverage.

This measures, for each held-out receptor, the highest sequence identity it
shares with any PDB entry released *before* the cutoff -- i.e. with the training
set. The search is a single RCSB query per receptor with the date filter ANDed
into the sequence query, so the top hit's identity is by construction the
maximum training homology; no separate date lookup is needed.

Section 7.10's conclusion is then re-tested on the subset that survives each
threshold. If it holds on the strict subset, the temporal split was sufficient.
If it does not, part of what Section 7.10 calls held-out was homology.

Usage:
    python src/homology_decontamination.py
"""

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

ART = REPO_ROOT / "artifacts"
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
CUTOFF = "2021-09-30"

# Thresholds in current use. 0.30 is ProteinArena's; 0.50 is the MMseqs2
# convention; 0.90 is the loosest reading under which a receptor could still be
# called unseen, and is the same threshold the panel uses for its own dedup.
TIERS = [("strict (<30%)", 0.30), ("moderate (<50%)", 0.50), ("loose (<90%)", 0.90)]


def max_training_identity(seq, retries=3):
    """Highest identity to any entry released before the cutoff, or 0.0."""
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "sequence",
             "parameters": {"evalue_cutoff": 10, "identity_cutoff": 0.0,
                            "sequence_type": "protein", "value": seq}},
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_accession_info.initial_release_date",
                            "operator": "less", "value": CUTOFF}},
        ]},
        "request_options": {"scoring_strategy": "sequence",
                            "results_verbosity": "verbose",
                            "paginate": {"start": 0, "rows": 25}},
        "return_type": "polymer_entity",
    }
    for attempt in range(retries):
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "90", "-X", "POST", SEARCH_URL,
             "-H", "Content-Type: application/json", "-d", json.dumps(query)],
            capture_output=True, text=True)
        try:
            data = json.loads(proc.stdout)
        except Exception:
            time.sleep(2 * (attempt + 1))
            continue
        best, who = 0.0, None
        for hit in data.get("result_set", []):
            for svc in hit.get("services", []):
                for node in svc.get("nodes", []):
                    for mc in node.get("match_context", []):
                        ident = mc.get("sequence_identity")
                        if ident is not None and ident > best:
                            best, who = float(ident), hit["identifier"]
        return best, who, data.get("total_count", 0)
    return 0.0, None, 0


def survey(ids, seqdir):
    out = {}
    print(f"{'PDB':6} {'max identity to pre-cutoff PDB':>31} {'closest':>10} {'hits':>6}")
    print("-" * 60)
    for pid in ids:
        f = seqdir / f"{pid}.json"
        if not f.exists():
            continue
        seq = json.loads(f.read_text())["receptor"]
        ident, who, n = max_training_identity(seq)
        out[pid] = {"max_identity": ident, "closest": who, "n_hits": n}
        flag = "  <-- effectively seen" if ident >= 0.90 else ""
        print(f"{pid:6} {ident:31.3f} {str(who or '-'):>10} {n:6d}{flag}")
        time.sleep(0.4)
    return out


def effect(recs, keep, metric):
    recs = [r for r in recs if r["receptor_id"] in keep
            and r.get("peptide_from", r["receptor_id"]) in keep]
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    diffs = []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        s = [x for x in g if x["label"] == "scrambled"]
        if c and s:
            diffs += [c[0][metric] - x[metric] for x in s]
    if len(diffs) < 3:
        return None
    d = np.array(diffs, float)
    return {"n_receptors": len(by), "effect": float(d.mean()),
            "p": float(stats.ttest_1samp(d, 0).pvalue)}


def rank_test(recs, keep, metric):
    recs = [r for r in recs if r["receptor_id"] in keep
            and r.get("peptide_from", r["receptor_id"]) in keep]
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    ranks, sizes = [], []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        d = [x for x in g if x["label"] == "decoy"]
        if not c or not d:
            continue
        sc = [c[0][metric]] + [x[metric] for x in d]
        ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
        sizes.append(len(sc))
    if len(ranks) < 4:
        return None
    r = np.array(ranks, float)
    exp = (np.array(sizes, float) + 1) / 2
    p = stats.wilcoxon(r - exp)[1] if not np.allclose(r - exp, 0) else 1.0
    return {"n": len(ranks), "mean_rank": float(r.mean()),
            "chance": float(exp.mean()), "p": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "homology_decontamination.json"))
    ap.add_argument("--cache", action="store_true",
                    help="reuse a previous survey instead of re-querying RCSB")
    args = ap.parse_args()

    from heldout_panel import ALREADY, NEW
    ids = ALREADY + NEW
    cache_p = Path(args.out)
    if args.cache and cache_p.exists():
        surveyed = json.loads(cache_p.read_text())["survey"]
        print(f"reusing cached survey of {len(surveyed)} receptors")
    else:
        print(f"Held-out panel: identity to the PDB as it stood before {CUTOFF}\n")
        surveyed = survey(ids, ART / "heldout_panel" / "sequences")

    print(f"\n{'=' * 72}\nHow much of the 'held-out' panel is actually novel?\n{'=' * 72}")
    vals = np.array([v["max_identity"] for v in surveyed.values()])
    print(f"  median max identity to training-era PDB: {np.median(vals):.3f}")
    for label, thr in TIERS:
        keep = {k for k, v in surveyed.items() if v["max_identity"] < thr}
        print(f"  {label:18} {len(keep):2d}/{len(surveyed)} receptors")

    recs = json.loads((ART / "heldout_panel" / "heldout_scores.json").read_text())
    print(f"\n{'=' * 72}\nSection 7.10 re-tested on each subset\n{'=' * 72}")
    print(f"{'subset':20} {'metric':14} {'n':>3} {'effect':>9} {'p':>9} "
          f"{'rank':>6} {'p':>8}")
    print("-" * 72)
    out = {"survey": surveyed, "subsets": {}}
    tiers = [("all (temporal only)", 1.01)] + TIERS
    for label, thr in tiers:
        keep = {k for k, v in surveyed.items() if v["max_identity"] < thr}
        for metric in ("iptm", "iface_plddt"):
            e = effect(recs, keep, metric)
            r = rank_test(recs, keep, metric)
            if not e:
                continue
            out["subsets"].setdefault(label, {})[metric] = {"effect": e, "rank": r}
            rk = f"{r['mean_rank']:6.2f} {r['p']:8.4f}" if r else f"{'-':>6} {'-':>8}"
            print(f"{label:20} {metric:14} {e['n_receptors']:3d} "
                  f"{e['effect']:+9.3f} {e['p']:9.5f} {rk}")
        print()

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
