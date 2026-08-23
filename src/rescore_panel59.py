"""Add interface pLDDT to the 59-receptor panel.

pdb_binder_benchmark.py scores one rank-key (ipTM). Section 8.2 recommends
interface pLDDT, and Section 7.6 shows it carries 8.6x ipTM's effect-to-noise on
the scramble control, so the panel is not usable for the paper's headline claim
without it. The structures are on disk, so this needs no folding.

The readout is Section 7.7's, byte-identical to the one used on Boltz-2's panel
and on Chai-1's arm: 8 A between representative atoms, CB where present and CA
otherwise, mean CA pLDDT over the contacting residues of both sides.

Usage:
    python src/rescore_panel59.py --work artifacts/panel59 --out artifacts/panel59_readouts.json
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]


def readouts(path, cutoff=8.0):
    import gemmi
    import numpy as np
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    if len(st) == 0 or len(st[0]) < 2:
        return None
    chains = sorted(st[0], key=len)
    pep, rec = chains[0], chains[-1]

    def rep(r):
        return r.find_atom("CB", "*") or r.find_atom("CA", "*")

    pr = [(r, rep(r)) for r in pep]
    rr = [(r, rep(r)) for r in rec]
    pr = [(r, a) for r, a in pr if a is not None]
    rr = [(r, a) for r, a in rr if a is not None]
    pi, ri = set(), set()
    for i, (_, a1) in enumerate(pr):
        for j, (_, a2) in enumerate(rr):
            if a1.pos.dist(a2.pos) <= cutoff:
                pi.add(i); ri.add(j)
    if not pi or not ri:
        return None

    def b(res):
        a = res.find_atom("CA", "*")
        return a.b_iso if a is not None else None

    pv = [b(pr[i][0]) for i in pi]
    rv = [b(rr[j][0]) for j in ri]
    pv = [v for v in pv if v is not None]
    rv = [v for v in rv if v is not None]
    if not pv or not rv:
        return None
    return {"iface_plddt": float(np.mean(pv + rv)),
            "receptor_side": float(np.mean(rv)),
            "peptide_side": float(np.mean(pv)),
            "n_pep_iface": len(pi), "n_rec_iface": len(ri)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(REPO_ROOT / "artifacts" / "panel59"))
    ap.add_argument("--scores", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "panel59_readouts.json"))
    args = ap.parse_args()

    work = Path(args.work)
    scores = json.loads(Path(args.scores or work / "pdb_binder_scores.json").read_text())
    recs = scores if isinstance(scores, list) else scores.get("per_fold", [])
    print(f"scored folds in manifest: {len(recs)}")

    # Only the benchmark's own batch_* folds; the msa_fetch probes are 5-step
    # throwaways and must not be mixed in.
    found = {}
    for p in work.glob("batch_*/**/predictions/**/*.pdb"):
        found.setdefault(p.parent.name, p)
    print(f"structures under batch_*: {len(found)}")

    out, missing = [], 0
    for r in recs:
        p = found.get(r["name"])
        if p is None:
            missing += 1
            continue
        v = readouts(p)
        if v is None:
            missing += 1
            continue
        out.append({**{k: r[k] for k in ("receptor_id", "label", "name", "peptide_from")},
                    "iptm": r.get("score"), **v})

    Path(args.out).write_text(json.dumps({"per_fold": out}, indent=1))
    print(f"rescored {len(out)} folds, {missing} without a usable structure")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
