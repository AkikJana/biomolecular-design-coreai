"""A panel the model was never trained on, and the same tests run against it.

Boltz-1 was trained on PDB entries released before 2021-09-30 (the AlphaFold3
cutoff); DeCAF distils Boltz-1 and inherits it. Of the original 22-receptor
panel only 6 postdate that cutoff, and on those 6 the DeCAF cognate-vs-scramble
effect fell from +0.265 to +0.030 for ipTM and from +12.03 to +2.91 for
interface pLDDT, with a significant interaction (p ~ 0.03). Six receptors is far
too few to conclude from.

This builds a second panel of the same size drawn *entirely* from post-cutoff
entries, using the identical screen -- PTM-clean, tag-free, receptor and peptide
deduplicated. Decoys are drawn from within the held-out set, so no fold in the
comparison involves a structure the model was trained on.

If the effect holds here, Section 7.8 is measuring prediction. If it collapses,
much of it is retrieval.

Usage:
    python src/heldout_panel.py --batch-size 12
"""

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from decaf_scramble_test import fold  # noqa: E402
from interface_side_split import sides  # noqa: E402
from pdb_binder_benchmark import build_pairs, fetch_complex, fetch_receptor_msa  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DECAF_HOME = Path.home() / ".boltz" / "decaf"
CUTOFF = "2021-09-30"

# Post-cutoff receptors already folded in the main panel, plus those found by
# discover_pdb_binders.py --released-after 2021-09-30.
ALREADY = ["9F6S", "8KDX", "9GRF", "7S7J", "8HLO", "7OKL"]
NEW = ["8OP0", "7RUP", "8JJV", "8PEF", "7E0B", "7JZQ", "7F3S", "9C66",
       "8VGC", "7UHE", "7PVX", "7TZK", "8YTG", "8TFU", "7XV0", "8X8A"]

# Members whose peptide chain carries a covalently attached non-standard
# component, from discover_pdb_binders.cofactors(). The benchmark folds the
# canonical sequence, so for these the thing folded is not the thing that was
# crystallised -- the same defect that excluded 1I8H.
#   9GRF  A2G  GalNAc O-linked at Ser3/Thr4; StcE is a mucin-selective protease
#              and reads the glycan, so AASTTTPAPA alone is not a substrate
#   7F3S  LBZ  benzoyl-lysine in a histone H3 tail; the PTM filter's allowlist
#              did not contain this acylation, so the audit passed it
#   7JZQ  VU1  4-[(piperidin-1-yl)methyl]benzoic acid bonded to the peptide
# Kept rather than dropped -- dropping costs power and the flags are a judgement
# about biology, not a measurement -- and the headline is reported both ways.
COFACTOR_FLAGGED = {"9GRF": "A2G", "7F3S": "LBZ", "7JZQ": "VU1"}

# 9GRJ (StcE) and 9GHK (Fyn SH3) were in this list first and are not any more.
# discover_pdb_binders.py could not see 9GRF's and 8KDX's cached sequences, so
# it never compared against them; the pair identities were 0.98 and 0.96. Two
# receptors that similar make each other's cognate peptide a mislabelled decoy,
# which would have shrunk the very effect this panel exists to measure.
# audit_panel_duplicates() below re-checks the final list at run time rather
# than trusting the discovery script, at a stricter threshold than it uses.
RECEPTOR_MAX_ID = 0.80
PEPTIDE_MAX_ID = 0.60


def audit_panel_duplicates(complexes):
    """Refuse to fold a panel containing near-duplicate receptors or peptides."""
    from difflib import SequenceMatcher
    ids, bad = sorted(complexes), []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ra = SequenceMatcher(None, complexes[a]["receptor"],
                                 complexes[b]["receptor"]).ratio()
            pa = SequenceMatcher(None, complexes[a]["peptide"],
                                 complexes[b]["peptide"]).ratio()
            if ra >= RECEPTOR_MAX_ID or pa >= PEPTIDE_MAX_ID:
                bad.append(f"{a}/{b} receptor {ra:.2f} peptide {pa:.2f}")
    if bad:
        raise SystemExit("near-duplicate panel members, refusing to fold:\n  "
                         + "\n  ".join(bad))
    print(f"  duplicate audit clean ({len(ids)} receptors, "
          f"receptor < {RECEPTOR_MAX_ID}, peptide < {PEPTIDE_MAX_ID})")


def seed_msa_cache(work):
    """Copy the main panel's MSAs for receptors this panel shares with it.

    The six ALREADY receptors are folded in both panels. Re-fetching their
    alignments would give a different homolog set on a different day, so any
    held-out-versus-in-training difference would be partly an MSA difference.
    Reusing the cached CSV keeps those six byte-identical across the comparison.
    """
    dst = work / "msa_cache"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for rid in ALREADY:
        if (dst / f"{rid}.csv").exists():
            continue
        for src_dir in ("pdb_binders_b2_n22", "pdb_binders_b2", "pdb_binders"):
            src = REPO_ROOT / "artifacts" / src_dir / "msa_cache" / f"{rid}.csv"
            if src.exists():
                shutil.copy2(src, dst / f"{rid}.csv")
                n += 1
                break
    print(f"  reused {n} cached MSA(s) from the main panel")


def verify_post_cutoff(ids, cache=REPO_ROOT / "artifacts" / "pdb_release_dates.json"):
    """Refuse to build the panel if any entry predates the cutoff."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from contamination_and_promiscuity import release_dates
    dates = release_dates(ids, cache=cache)
    bad = {i: dates.get(i, "?") for i in ids if not dates.get(i, "") >= CUTOFF}
    if bad:
        raise SystemExit(f"not post-cutoff, refusing to proceed: {bad}")
    return dates


def score_dir(results, names, with_pae=False):
    parser = PDBParser(QUIET=True)
    out = {}
    for n in names:
        d = results / "predictions" / n
        pdb, conf = d / f"{n}_model_0.pdb", d / f"confidence_{n}_model_0.json"
        if not pdb.exists():
            continue
        try:
            model = parser.get_structure("x", str(pdb))[0]
            s = sides(model)
        except Exception:
            continue
        if not s:
            continue
        s["iptm"] = (json.loads(conf.read_text())["iptm"]
                     if conf.exists() else float("nan"))
        if with_pae:
            from pae_readouts import load_pae, readouts
            pae = load_pae(d, n)
            if pae is not None:
                r = readouts(model, pae)
                if r:
                    s.update(r)
        out[n] = s
    return out


# Lower PAE is better, so these are negated before every test; otherwise a
# working metric would look like it runs backwards.
FLIP = {"iface_pae", "mpae"}
PAE_METRICS = ("iface_pae", "mpae", "pae_frac_lt10", "ipsae", "pdockq2")


def metric_list(recs):
    base = ["iptm", "iface_plddt", "receptor_side"]
    return base + [m for m in PAE_METRICS if any(m in r for r in recs)]


def val(r, met):
    v = r[met]
    return -v if met in FLIP else v


def analyse(recs, drop=(), title="Held-out panel"):
    if drop:
        recs = [r for r in recs
                if r["receptor_id"] not in drop and r.get("peptide_from") not in drop]
    by = {}
    for r in recs:
        by.setdefault(r["receptor_id"], []).append(r)
    print(f"\n{'=' * 74}\n{title}: {len(recs)} folds, {len(by)} receptors"
          f"\n{'=' * 74}")
    print(f"\n{'metric':16} {'cognate':>9} {'scram':>9} {'cog-scr':>9}"
          f" {'95% CI':>20} {'p':>9}")
    print("-" * 74)
    out = {}
    for met in metric_list(recs):
        if not all(met in r for r in recs):
            continue
        cog = np.array([val(r, met) for r in recs if r["label"] == "cognate"])
        scr = np.array([val(r, met) for r in recs if r["label"] == "scrambled"])
        diffs = []
        for g in by.values():
            c = [x for x in g if x["label"] == "cognate"]
            s = [x for x in g if x["label"] == "scrambled"]
            if c and s:
                diffs += [val(c[0], met) - val(x, met) for x in s]
        if len(diffs) < 3:
            continue
        d = np.array(diffs, float)
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(),
                              scale=d.std(ddof=1) / np.sqrt(len(d)))
        p = stats.ttest_1samp(d, 0).pvalue
        out[met] = {"cognate": float(cog.mean()), "scrambled": float(scr.mean()),
                    "effect": float(d.mean()), "ci95": [float(ci[0]), float(ci[1])],
                    "p": float(p), "n_pairs": len(d), "n_receptors": len(by)}
        print(f"{met:16} {cog.mean():9.3f} {scr.mean():9.3f} {d.mean():+9.3f}"
              f" [{ci[0]:+7.3f},{ci[1]:+7.3f}] {p:9.5f}")

    print("\n  In-training reference (16 receptors, DeCAF):")
    print("    iptm         +0.265 (p 1e-5)   |  previous held-out (n=6): +0.030")
    print("    iface_plddt +12.027 (p <1e-5)  |  previous held-out (n=6): +2.909")

    # rank against decoys
    print(f"\n{'metric':16} {'#1':>4} {'mean rank':>10} {'chance':>7} {'p':>9}")
    for met in metric_list(recs):
        if not all(met in r for r in recs):
            continue
        ranks, sizes = [], []
        for g in by.values():
            c = [x for x in g if x["label"] == "cognate"]
            dd = [x for x in g if x["label"] == "decoy"]
            if not c or not dd:
                continue
            sc = [val(c[0], met)] + [val(x, met) for x in dd]
            ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
            sizes.append(len(sc))
        if len(ranks) < 5:
            continue
        r = np.array(ranks, float); s = np.array(sizes, float)
        exp = (s + 1) / 2
        p = (stats.wilcoxon(r - exp, alternative="two-sided")[1]
             if not np.allclose(r - exp, 0) else 1.0)
        out.setdefault(met, {}).update({"mean_rank": float(r.mean()),
                                        "chance": float(exp.mean()),
                                        "cognate_first": int((r == 1).sum()),
                                        "p_rank": float(p)})
        print(f"{met:16} {int((r == 1).sum()):4d} {r.mean():10.2f} {exp.mean():7.2f}"
              f" {p:9.4f}")
    print("  Main panel reference: ipTM 1.77 (p 0.0087), interface pLDDT 1.73 (p 0.0042)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=12)
    # Section 7.10 folded this panel at 10 steps, 1 recycling pass and MSA depth
    # 32 on DeCAF, and Section 7.13 then showed those three reductions suppress
    # the effect three- to sevenfold. Whether the held-out penalty survives at
    # the intended settings is therefore untested, and these flags are how it
    # gets tested. Defaults reproduce Section 7.10 exactly.
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--msa-depth", type=int, default=32,
                    help="rows to subsample the alignment to; 0 takes it whole")
    ap.add_argument("--base", default="decaf", choices=("decaf", "boltz1", "boltz2"),
                    help="decaf is Section 7.10's arm; boltz1 matches Section "
                         "7.13's full-settings arm, which is the like-for-like "
                         "comparison for a contamination penalty")
    ap.add_argument("--decoys", type=int, default=3)
    ap.add_argument("--scrambles", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=str(DECAF_HOME / "decaf_conf_ckpt.ckpt"))
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "heldout_panel"))
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--keep-structures", action="store_true",
                    help="do not delete each batch after scoring. Needed to "
                         "compare the *pose* across draws rather than the "
                         "score: a real binder should land in the same place "
                         "each time and a non-binder should scatter.")
    ap.add_argument("--extra", default=None,
                    help="JSON from discover_pdb_binders.py --released-after, "
                         "appended to the built-in post-cutoff panel. Every id is "
                         "still checked against the cutoff before folding.")
    ap.add_argument("--run-tag", default="",
                    help="suffix for the score store, so a repeat run is a new "
                         "independent draw rather than a no-op resume. Folds "
                         "are unseeded, and Section 7.10 was written off a "
                         "single draw whose interface-pLDDT p-value moved four "
                         "orders of magnitude on the second.")
    ap.add_argument("--with-pae", action="store_true",
                    help="write and score the full PAE matrix. Folds are "
                         "unseeded, so this is a second independent draw of the "
                         "whole panel rather than a rescore -- which also makes "
                         "it a replication of the Section 7.10 result.")
    ap.add_argument("--msa-only", action="store_true",
                    help="fetch and cache alignments, then stop -- lets the\nboltz1 checkpoint the fetch needs be deleted before DeCAF folding starts")
    args = ap.parse_args()

    ids = ALREADY + NEW
    if args.extra:
        raw = json.loads(Path(args.extra).read_text())
        items = raw if isinstance(raw, list) else next(iter(raw.values()))
        found = [e["pdb_id"] if isinstance(e, dict) else e for e in items]
        added = [i for i in found if i not in ids]
        ids = ids + added
        print(f"  {len(added)} receptors added from {args.extra}; "
              f"panel is now {len(ids)}")
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)
    dates = verify_post_cutoff(ids)
    print(f"{len(ids)} receptors, all released after {CUTOFF} "
          f"(earliest {min(dates[i] for i in ids)})")

    complexes = {}
    for pid in ids:
        rec, pep, rn, _ = fetch_complex(pid, work / "sequences")
        complexes[pid] = {"receptor": rec, "peptide": pep}
        print(f"  {pid} {dates[pid]}  receptor {len(rec):3d}aa ({rn[:28]:28}) "
              f"peptide {len(pep):2d}aa {pep}")

    audit_panel_duplicates(complexes)
    pairs = build_pairs(complexes, args.decoys, args.scrambles, args.seed)
    for i, p in enumerate(pairs):
        p["name"] = f"ho_{i:03d}"
    print(f"\n{len(pairs)} folds: "
          f"{sum(p['label'] == 'cognate' for p in pairs)} cognate, "
          f"{sum(p['label'] == 'decoy' for p in pairs)} decoy, "
          f"{sum(p['label'] == 'scrambled' for p in pairs)} scrambled")

    store = work / (f"heldout_scores{'_pae' if args.with_pae else ''}"
                    f"{args.run_tag}.json")
    recs = json.loads(store.read_text()) if store.exists() else []

    # The scores file is a bare list that three other scripts read, so the
    # settings go in a sidecar rather than changing its shape. Without this a
    # reduced-settings and a full-settings run are indistinguishable on disk,
    # which is exactly the confusion Section 7.13 exists to resolve.
    (work / f"heldout_settings{args.run_tag}.json").write_text(json.dumps({
        "base": args.base, "sampling_steps": args.sampling_steps,
        "recycling_steps": args.recycling_steps,
        "msa_depth": args.msa_depth or "full",
        "decoys": args.decoys, "scrambles": args.scrambles, "seed": args.seed,
        "run_tag": args.run_tag or "(none)", "n_receptors": len(ids),
    }, indent=2))
    if not args.analyse_only:
        seed_msa_cache(work)
        msas = {}
        for rid in complexes:
            msas[rid] = fetch_receptor_msa(work, rid, complexes[rid]["receptor"],
                                           complexes[rid]["peptide"], "boltz1")
        # The probe folds exist only to make the server return an alignment; the
        # alignments are cached, and the structures are throwaway. Deleting them
        # matters here because MSA fetch needs the 3.3 GB boltz1 checkpoint that
        # the DeCAF folding below does not, on a volume with little room spare.
        shutil.rmtree(work / "msa_fetch", ignore_errors=True)
        if args.msa_only:
            print(f"\nMSAs cached in {work / 'msa_cache'}; stopping before folding.")
            return
        done = {r["name"] for r in recs}
        todo = [p for p in pairs if p["name"] not in done]
        for start in range(0, len(todo), args.batch_size):
            chunk = todo[start:start + args.batch_size]
            # The run tag has to be in the batch directory name, not just the
            # score file: with --keep-structures two runs would otherwise write
            # predictions into the same b00/b01 and the second would overwrite
            # the first, which is exactly the comparison being set up.
            bdir = work / f"b{args.run_tag}{start // args.batch_size:02d}"
            inputs = bdir / "inputs"
            if inputs.exists():
                shutil.rmtree(inputs)
            inputs.mkdir(parents=True)
            for p in chunk:
                msa = msas.get(p["receptor_id"])
                rline = f"      msa: {msa}\n" if msa else "      msa: empty\n"
                (inputs / f"{p['name']}.yaml").write_text(
                    "version: 1\nsequences:\n"
                    f"  - protein:\n      id: A\n      sequence: {p['receptor']}\n{rline}"
                    f"  - protein:\n      id: B\n      sequence: {p['peptide']}\n"
                    f"      msa: empty\n")
            res, el = fold(inputs, bdir, args.ckpt, args.sampling_steps,
                           args.recycling_steps, args.base,
                           write_pae=args.with_pae, msa_depth=args.msa_depth)
            got = score_dir(res, [p["name"] for p in chunk],
                            with_pae=args.with_pae)
            for p in chunk:
                if p["name"] in got:
                    recs.append({**{k: p[k] for k in
                                    ("name", "receptor_id", "label", "peptide_from")},
                                 **got[p["name"]]})
            store.write_text(json.dumps(recs, indent=2))
            print(f"  batch {start // args.batch_size}: {len(got)}/{len(chunk)} "
                  f"in {el:.0f}s", flush=True)
            if not args.keep_structures:
                shutil.rmtree(bdir, ignore_errors=True)

    summary = analyse(recs)
    # Sensitivity: the same test with the cofactor-dependent members removed,
    # both as receptors and as decoy donors. If the two agree, the flags do not
    # drive the result; if they disagree, the flagged members do and the
    # restricted number is the one to quote.
    sens = analyse(recs, drop=set(COFACTOR_FLAGGED),
                   title="Sensitivity, cofactor-flagged members removed")
    tag = ("_pae" if args.with_pae else "") + args.run_tag
    (REPO_ROOT / "artifacts" / f"heldout_panel_result{tag}.json").write_text(
        json.dumps({"per_fold": recs, "summary": summary,
                    "sensitivity_no_cofactor": sens,
                    "cofactor_flagged": COFACTOR_FLAGGED, "dates": dates}, indent=2))
    print(f"\nwrote {REPO_ROOT / 'artifacts'}/heldout_panel_result{tag}.json")


if __name__ == "__main__":
    main()
