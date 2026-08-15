"""Does the peptide land in the groove the crystal says it should?

Every readout in this dissertation is the model's opinion of its own output --
ipTM, pLDDT, PAE, pDockQ all come from the confidence head. That is why they all
sit under the variance ceiling of Section 7.9.2, and why combining them, or
combining whole models, buys nothing.

This uses information from outside the model. The crystal structure says which
receptor residues actually contact the peptide. For any predicted complex --
cognate, decoy or scramble -- one can then ask what fraction of the predicted
contacts fall inside that known site. A decoy can be placed confidently and
still be placed in the wrong place; confidence cannot see that, and this can.

Nothing in Section 7.9's decomposition constrains it, because the reference is
not something the model reported.

It is also deployable rather than an oracle trick. In a screen you do not know
whether a candidate binds -- that is the question -- but you almost always know
where your target's binding site is, from a known complex or from annotation.
The site is a property of the *receptor*, and it is held fixed across every
candidate scored against it.

Method. The crystal's true site is the set of receptor residues with any heavy
atom within 5 A of any peptide heavy atom. Crystal numbering is author-assigned
and gapped, while the folded structure is a clean 1..N of the canonical
sequence, so the two are aligned by sequence and the site indices mapped across.
Predicted contacts use the 8 A CB-CB criterion used everywhere else in this work.

  site precision  fraction of predicted receptor contacts inside the true site
  site recall     fraction of true-site residues that were contacted

Requires structures kept from a fold run (`--keep-structures`), and a model
whose backbones are physical -- Section 7.11 shows that at ten sampling steps a
stock model's are not, which would make any contact-based measure meaningless.

Usage:
    python src/site_correctness.py --tag p1
"""

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
ART = REPO_ROOT / "artifacts"
PANEL = ART / "heldout_panel"
CIF_DIR = ART / "crystals"
warnings.filterwarnings("ignore")

SITE_CUTOFF = 5.0      # heavy-atom, crystal
CONTACT_CUTOFF = 8.0   # CB-CB, matching the rest of this work


def three_to_one(res):
    try:
        return protein_letters_3to1[res.get_resname()]
    except KeyError:
        return "X"


def fetch_cif(pdb_id):
    CIF_DIR.mkdir(parents=True, exist_ok=True)
    path = CIF_DIR / f"{pdb_id}.cif"
    if path.exists() and path.stat().st_size > 0:
        return path
    for src in (ART / "cofactor_check" / f"{pdb_id}.cif",
                ART / "pdb_candidates" / f"{pdb_id}.cif"):
        if src.exists() and src.stat().st_size > 0:
            path.write_bytes(src.read_bytes())
            return path
    subprocess.run(["curl", "-s", "--max-time", "60",
                    f"https://files.rcsb.org/download/{pdb_id}.cif",
                    "-o", str(path)], capture_output=True)
    return path if path.exists() and path.stat().st_size > 0 else None


def crystal_site(pdb_id):
    """(receptor sequence, set of 0-based site indices) from the crystal."""
    path = fetch_cif(pdb_id)
    if path is None:
        return None
    try:
        model = MMCIFParser(QUIET=True).get_structure("x", str(path))[0]
    except Exception:
        return None
    chains = []
    for ch in model:
        res = [r for r in ch if r.id[0] == " " and "CA" in r]
        if len(res) >= 4:
            chains.append(res)
    if len(chains) < 2:
        return None
    chains.sort(key=len)
    pep, rec = chains[0], chains[-1]
    pep_atoms = np.array([a.coord for r in pep for a in r
                          if a.element != "H"], dtype=float)
    if not len(pep_atoms):
        return None
    site = set()
    for i, r in enumerate(rec):
        atoms = np.array([a.coord for a in r if a.element != "H"], dtype=float)
        if not len(atoms):
            continue
        d = np.linalg.norm(atoms[:, None, :] - pep_atoms[None, :, :], axis=-1)
        if d.min() < SITE_CUTOFF:
            site.add(i)
    seq = "".join(three_to_one(r) for r in rec)
    return seq, site


def map_site(crystal_seq, site_idx, folded_seq):
    """Carry crystal site indices onto the folded numbering by alignment.

    The crystal is missing disordered residues and uses author numbering; the
    folded chain is a clean 1..N of the canonical sequence. Aligning the two
    is what makes the site comparable at all.
    """
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.match_score = 2
    aligner.mismatch_score = -1
    try:
        aln = aligner.align(crystal_seq, folded_seq)[0]
    except Exception:
        return None
    mapped = set()
    for (cs, ce), (fs, fe) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(ce - cs):
            if (cs + k) in site_idx:
                mapped.add(fs + k)
    return mapped


def predicted_contacts(path, parser):
    """(receptor contact indices, receptor length, folded receptor sequence)."""
    model = parser.get_structure("x", str(path))[0]
    chains = []
    for ch in model:
        res = [r for r in ch if "CA" in r]
        if res:
            chains.append(res)
    if len(chains) != 2:
        return None
    chains.sort(key=len)
    pep, rec = chains[0], chains[-1]
    cb = lambda rs: np.array([(r["CB"].coord if "CB" in r else r["CA"].coord)
                              for r in rs], dtype=float)
    d = np.linalg.norm(cb(rec)[:, None, :] - cb(pep)[None, :, :], axis=-1)
    idx = set(np.where(d.min(axis=1) < CONTACT_CUTOFF)[0].tolist())
    return idx, len(rec), "".join(three_to_one(r) for r in rec)


def evaluate(rows, metric, label):
    by = {}
    for r in rows:
        by.setdefault(r["receptor_id"], []).append(r)
    diffs = []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        s = [x for x in g if x["label"] == "scrambled"]
        if c and s:
            diffs += [c[0][metric] - x[metric] for x in s]
    ranks, sizes = [], []
    for g in by.values():
        c = [x for x in g if x["label"] == "cognate"]
        d = [x for x in g if x["label"] == "decoy"]
        if not c or not d:
            continue
        sc = [c[0][metric]] + [x[metric] for x in d]
        ranks.append(1 + sum(v >= sc[0] for v in sc[1:]))
        sizes.append(len(sc))
    e = np.array(diffs, float)
    r = np.array(ranks, float)
    exp = (np.array(sizes, float) + 1) / 2
    out = {"effect": float(e.mean()), "p": float(stats.ttest_1samp(e, 0).pvalue),
           "mean_rank": float(r.mean()), "chance": float(exp.mean()),
           "first": int((r == 1).sum()), "n_receptors": len(r),
           "rank_p": float(stats.wilcoxon(r - exp)[1])}
    print(f"{label:34} {out['effect']:+9.3f} {out['p']:9.5f} "
          f"{out['mean_rank']:6.2f} {out['rank_p']:8.4f} "
          f"{out['first']:3d}/{out['n_receptors']:<3d}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p1", help="run tag whose structures to score")
    ap.add_argument("--out", default=str(ART / "site_correctness.json"))
    args = ap.parse_args()

    meta = {}
    for f in PANEL.glob("heldout_scores*.json"):
        for r in json.loads(f.read_text()):
            meta.setdefault(r["name"], r)

    sites = {}
    from heldout_panel import ALREADY, NEW
    for pid in ALREADY + NEW:
        got = crystal_site(pid)
        if got:
            sites[pid] = got
    print(f"crystal sites resolved for {len(sites)}/{len(ALREADY + NEW)} receptors")
    sz = [len(s) for _, s in sites.values()]
    print(f"  true site size: median {np.median(sz):.0f} residues "
          f"(range {min(sz)}-{max(sz)})\n")

    parser = PDBParser(QUIET=True)
    rows, mapped_cache = [], {}
    files = sorted(PANEL.glob(f"b{args.tag}*/boltz_results_inputs/predictions/*/*_model_0.pdb"))
    for path in files:
        name = path.parent.name
        m = meta.get(name)
        if not m or m["receptor_id"] not in sites:
            continue
        got = predicted_contacts(path, parser)
        if not got:
            continue
        contacts, n_rec, folded_seq = got
        rid = m["receptor_id"]
        if rid not in mapped_cache:
            cseq, cidx = sites[rid]
            mapped_cache[rid] = map_site(cseq, cidx, folded_seq)
        true_site = mapped_cache[rid]
        if not true_site or not contacts:
            continue
        hit = len(contacts & true_site)
        rows.append({"name": name, "receptor_id": rid, "label": m["label"],
                     "site_precision": hit / len(contacts),
                     "site_recall": hit / len(true_site),
                     "n_contacts": len(contacts), "n_site": len(true_site),
                     "iface_plddt": m.get("iface_plddt"), "iptm": m.get("iptm")})
    print(f"{len(rows)} folds scored, {len({r['receptor_id'] for r in rows})} receptors\n")

    for lab in ("cognate", "decoy", "scrambled"):
        v = [r for r in rows if r["label"] == lab]
        if v:
            print(f"  {lab:10} site precision {np.mean([x['site_precision'] for x in v]):.3f}"
                  f"   recall {np.mean([x['site_recall'] for x in v]):.3f}"
                  f"   contacts {np.mean([x['n_contacts'] for x in v]):5.1f}")

    print(f"\n{'readout':34} {'cog-scr':>9} {'p':>9} {'rank':>6} {'p':>8} {'#1':>7}")
    print("-" * 78)
    res = {}
    for m, lab in (("site_precision", "site precision (vs crystal)"),
                   ("site_recall", "site recall (vs crystal)"),
                   ("iface_plddt", "interface pLDDT (reference)"),
                   ("iptm", "ipTM (reference)")):
        if all(r.get(m) is not None for r in rows):
            res[m] = evaluate(rows, m, lab)

    Path(args.out).write_text(json.dumps(
        {"per_fold": rows, "summary": res, "tag": args.tag}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
