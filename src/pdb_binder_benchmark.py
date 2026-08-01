"""Does ipTM separate real peptide binders from non-binders?

Every earlier benchmark here ranked synthetic peptides that were never designed
or evolved to bind anything, and the scramble control showed ipTM was insensitive
to binder sequence order. That leaves the central question unanswered: does the
reference work *at all* on genuine binders?

This benchmark uses experimentally determined peptide-domain complexes from the
PDB, so every positive is a pair known to bind. Sequences are fetched live from
RCSB and cached -- never transcribed by hand, since a wrong sequence would
silently invalidate the whole comparison.

Three classes, all folded identically:

  cognate    receptor + its own peptide, from the same PDB entry (true positive)
  decoy      receptor + a real peptide from a *different* complex. The strongest
             negative: a genuine binder, wrong partner. Screening is exactly this
             discrimination.
  scrambled  receptor + its own peptide with composition preserved and order
             destroyed (tests order sensitivity, as before)

If ipTM cannot rank cognate above decoy, it is not usable as a binder-ranking
reference, and no surrogate distilled from it can be either.

Usage:
    python src/pdb_binder_benchmark.py --decoys-per-receptor 3 --scrambles-per-receptor 2
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np

from boltz2_predict import BoltzCliPredictFn
from run_reference_benchmark import REPO_ROOT, run_boltz

# Peptide-domain complexes with a short peptide and a domain small enough to fold
# on CPU. Screened from RCSB for exactly two polymer chains, peptide <= 25 aa and
# receptor <= 140 aa.
#
# 22 receptors, sized by a power analysis on the Boltz-2 n=11 result: 80% power
# on the specificity test needs 21 (dz = 0.57). At n = 11 the test could not
# resolve the effect either way, so the honest response was more receptors
# rather than a different statistic.
#
# Selected by src/discover_pdb_binders.py, which applies this file's own screen
# programmatically. Three filters exist because earlier passes got them wrong:
#
#   peptide near-identity  decoys are built by giving a receptor another
#                          entry's peptide, so homologous peptides across
#                          receptors make "decoys" that genuinely bind.
#                          Rejected three histone H3 tails and a duplicated
#                          PI3K phosphopeptide.
#   tag / linker           4W8H offered "HHHHHH" -- a His-tag crystallised in
#                          a groove is not a binder.
#   PTM dependence         RCSB FASTA is canonical, so phosphoserine reads as S
#                          and acetyl-lysine as K. An SH2 domain binds pTyr and
#                          a bromodomain reads acetyl-lysine; folding the
#                          canonical peptide gives a cognate that cannot bind.
#
# The PTM screen removed 1I8H from the original panel -- Pin1 WW reads
# pSer/pThr-Pro, and 1I8H is exactly the receptor whose "true binder ranked
# last of six" under Boltz-1. Ranking a non-binder last is correct behaviour.
# See src/audit_panel_ptms.py.
#
# 3JQL and 4Z2O passed every automated filter and were dropped by hand: PLA2
# with an amyloid-beta hexapeptide is an incidental co-crystal, and
# hoefavidin's "hoef-peptide" is a self-fragment. Neither is a binding pair.
#
# Shared *folds* are kept on purpose -- discriminating peptides that bind
# similar domains is the task.
PDB_IDS = ["1YCR", "1CKA", "1BE9", "1SEM", "1ELW", "2GBQ", "1D4T", "1TP5",
           "2FNT", "1NLO", "1OAI", "4LN2", "9F6S", "8KDX", "6YOO", "3DS4",
           "9GRF", "7S7J", "4Z8J", "8HLO", "7OKL", "4O36"]


def fetch_complex(pdb_id: str, cache_dir: Path):
    """(receptor, peptide, receptor_name, peptide_name) from RCSB, cached."""
    cache = cache_dir / f"{pdb_id}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["receptor"], d["peptide"], d["receptor_name"], d["peptide_name"]
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    text = subprocess.run(["curl", "-s", "--max-time", "30", url],
                          capture_output=True, text=True).stdout.strip()
    lines = [ln for ln in text.split("\n") if ln]
    if not lines or not lines[0].startswith(">"):
        raise RuntimeError(f"could not fetch {pdb_id}")
    chains = []
    for i in range(0, len(lines) - 1, 2):
        parts = lines[i].split("|")
        chains.append((len(lines[i + 1]), lines[i + 1],
                       parts[2][:30] if len(parts) > 2 else "?"))
    chains.sort()
    if len(chains) != 2:
        raise RuntimeError(f"{pdb_id}: expected 2 chains, got {len(chains)}")
    pep, rec = chains[0], chains[1]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"receptor": rec[1], "peptide": pep[1],
                                 "receptor_name": rec[2], "peptide_name": pep[2]}))
    return rec[1], pep[1], rec[2], pep[2]


def build_pairs(complexes, decoys_per, scrambles_per, seed):
    """cognate / decoy / scrambled pairs, labelled."""
    rng = random.Random(seed)
    ids = list(complexes)
    pairs = []
    for rid in ids:
        rec = complexes[rid]["receptor"]
        pairs.append({"receptor_id": rid, "receptor": rec,
                      "peptide": complexes[rid]["peptide"], "label": "cognate",
                      "peptide_from": rid})
        others = [o for o in ids if o != rid]
        for did in rng.sample(others, min(decoys_per, len(others))):
            pairs.append({"receptor_id": rid, "receptor": rec,
                          "peptide": complexes[did]["peptide"], "label": "decoy",
                          "peptide_from": did})
        for _ in range(scrambles_per):
            s = list(complexes[rid]["peptide"])
            rng.shuffle(s)
            pairs.append({"receptor_id": rid, "receptor": rec,
                          "peptide": "".join(s), "label": "scrambled",
                          "peptide_from": rid})
    return pairs


def write_yaml(path: Path, receptor: str, peptide: str, receptor_msa):
    rline = f"      msa: {receptor_msa}\n" if receptor_msa else "      msa: empty\n"
    path.write_text(
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {receptor}\n{rline}"
        f"  - protein:\n      id: B\n      sequence: {peptide}\n      msa: empty\n"
    )


def fetch_receptor_msa(work: Path, rid: str, receptor: str, peptide: str,
                       model: str = "boltz1"):
    cache = work / "msa_cache" / f"{rid}.csv"
    if cache.exists():
        return str(cache)
    probe = work / "msa_fetch" / rid / "inputs"
    probe.mkdir(parents=True, exist_ok=True)
    # The receptor's msa: key must be OMITTED entirely for --use_msa_server to
    # fetch an alignment. Passing None to write_yaml emits "msa: empty", which
    # explicitly disables the fetch -- that silently produced single-sequence
    # folding for every receptor on the first attempt.
    (probe / "pair_000.yaml").write_text(
        "version: 1\nsequences:\n"
        f"  - protein:\n      id: A\n      sequence: {receptor}\n"
        f"  - protein:\n      id: B\n      sequence: {peptide}\n      msa: empty\n"
    )
    try:
        results, _ = run_boltz(probe, probe.parent, 0, 5, model,
                               use_msa=True, max_msa_seqs=32)
    except RuntimeError as exc:
        print(f"  [msa] {rid}: fetch failed ({str(exc)[:60]}); single-sequence")
        return None
    src = results / "msa" / "pair_000_0.csv"
    if not src.exists():
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(src.read_text())
    print(f"  [msa] {rid}: {len(src.read_text().splitlines()) - 1} homologs", flush=True)
    return str(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoys-per-receptor", type=int, default=3)
    ap.add_argument("--scrambles-per-receptor", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--recycling-steps", type=int, default=1)
    ap.add_argument("--rank-key", default="iptm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-msa", action="store_true")
    ap.add_argument("--model", default="boltz1", choices=["boltz1", "boltz2"],
                    help="Boltz-2 is a different model with its own weights; "
                         "results are not comparable across this flag unless "
                         "everything else is held fixed")
    ap.add_argument("--work-dir", default=str(REPO_ROOT / "artifacts" / "pdb_binders"))
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    complexes = {}
    for pid in PDB_IDS:
        rec, pep, rn, pn = fetch_complex(pid, work / "sequences")
        complexes[pid] = {"receptor": rec, "peptide": pep}
        print(f"  {pid}: receptor {len(rec)}aa ({rn}) | peptide {len(pep)}aa ({pn}) {pep}")

    pairs = build_pairs(complexes, args.decoys_per_receptor,
                        args.scrambles_per_receptor, args.seed)
    for i, p in enumerate(pairs):
        p["name"] = f"pair_{i:03d}"
    print(f"\n{len(pairs)} complexes to fold: "
          f"{sum(p['label']=='cognate' for p in pairs)} cognate, "
          f"{sum(p['label']=='decoy' for p in pairs)} decoy, "
          f"{sum(p['label']=='scrambled' for p in pairs)} scrambled")

    msas = {}
    if not args.skip_predict and not args.no_msa:
        for rid in complexes:
            msas[rid] = fetch_receptor_msa(work, rid, complexes[rid]["receptor"],
                                           complexes[rid]["peptide"])

    dirs = []
    if not args.skip_predict:
        for start in range(0, len(pairs), args.batch_size):
            chunk = pairs[start:start + args.batch_size]
            bdir = work / f"batch_{start // args.batch_size:02d}"
            idir = bdir / "inputs"
            idir.mkdir(parents=True, exist_ok=True)
            for p in chunk:
                write_yaml(idir / f"{p['name']}.yaml", p["receptor"], p["peptide"],
                           msas.get(p["receptor_id"]))
            res, el = run_boltz(idir, bdir, args.recycling_steps, args.sampling_steps,
                                args.model, max_msa_seqs=32)
            dirs.append(res)
            print(f"  batch {start // args.batch_size}: {len(chunk)} in {el:.0f}s", flush=True)
    else:
        dirs = sorted(work.glob("batch_*/boltz_results_inputs"))

    for p in pairs:
        p["score"] = float("nan")
        for d in dirs:
            try:
                p["score"] = BoltzCliPredictFn(str(d), lambda t, b: p["name"])(
                    "", "")[args.rank_key].reshape(-1)[0].item()
                break
            except (FileNotFoundError, KeyError):
                continue

    (work / "pdb_binder_scores.json").write_text(json.dumps(pairs, indent=2))
    report(pairs, args.rank_key)


def report(pairs, rank_key):
    from scipy import stats
    arr = {k: np.array([p["score"] for p in pairs
                        if p["label"] == k and not np.isnan(p["score"])])
           for k in ("cognate", "decoy", "scrambled")}
    print("\n" + "=" * 66)
    print(f"Can {rank_key} separate real binders from non-binders?")
    print("=" * 66)
    for k, v in arr.items():
        if len(v):
            print(f"  {k:10} n={len(v):3d}  mean {v.mean():.4f}  sd {v.std():.4f}  "
                  f"range {v.min():.4f}-{v.max():.4f}")

    def compare(a, b, la, lb):
        if len(a) < 3 or len(b) < 3:
            print(f"  {la} vs {lb}: too few")
            return
        u, p = stats.mannwhitneyu(a, b, alternative="greater")
        auc = u / (len(a) * len(b))          # P(cognate ranked above other)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / pooled if pooled > 0 else float("nan")
        verdict = "SEPARATES" if (p < 0.05 and auc > 0.5) else "does NOT separate"
        print(f"  {la} > {lb}:  AUC {auc:.3f}  Cohen's d {d:+.2f}  p {p:.4f}  -> {verdict}")

    print()
    compare(arr["cognate"], arr["decoy"], "cognate", "decoy    ")
    compare(arr["cognate"], arr["scrambled"], "cognate", "scrambled")
    print("\n  AUC 0.5 = no discrimination. For a usable screening reference you")
    print("  want cognate > decoy with AUC well above 0.5.")


if __name__ == "__main__":
    main()
