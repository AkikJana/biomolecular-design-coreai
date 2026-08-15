"""Find additional peptide-domain complexes for the binder benchmark.

The original 11-receptor panel was screened from RCSB by hand-checked criteria:
exactly two polymer chains, peptide <= 25 aa, receptor <= 140 aa. Extending the
panel to the ~21 receptors the power analysis calls for needs the same screen
applied programmatically, so the additions are selected by stated rules rather
than recalled from memory -- a wrong or mis-chained PDB ID would silently
corrupt the comparison.

Screen:
  * exactly 2 protein polymer entities, protein-only entry
  * X-ray, resolution <= 2.5 A
  * total deposited polymer length 60-190 residues
  * after fetching FASTA: peptide 6-25 aa, receptor 40-140 aa
  * receptor not >=90% identical to one already accepted or in the panel
  * peptide not >=60% identical to one already accepted or in the panel
  * peptide carries no binding-critical post-translational modification
  * peptide is not a purification tag or linker

**The peptide filter is the one that matters.** Decoys are built by handing a
receptor a peptide from a different entry, so if two receptors bind homologous
peptides that "decoy" is a genuine binder -- a mislabelled negative. A first
pass deduplicating only receptors accepted 5GJI and 5AUL (identical peptide
SDYMNMTP) and three separate histone H3 tail peptides (ARTKQTARKSTGGKA /
ARTKQTARKST / ARTKQTAAKA), any pair of which would have poisoned the decoy
class for the others.

**The PTM filter is the one that was missed first.** RCSB FASTA returns
canonical sequence, so phosphoserine reads as S and acetyl-lysine as K. An SH2
domain binds phosphotyrosine and a bromodomain reads acetyl-lysine; folding the
canonical peptide gives a cognate pair that does not bind. Seven of a first
25-receptor panel were PTM-dependent, including 1I8H from the *original* eleven
-- which is the receptor whose "true binder ranked last of six" in the Boltz-1
write-up. Ranking a non-binder last is correct behaviour, not a failure.

Deliberately NOT deduplicated by fold or family: SH3 and PDZ domains recur in
the panel on purpose, because a screening reference has to tell apart peptides
binding *similar* domains. Removing them would make the benchmark easier than
the task it stands in for. Sequence-level near-duplicates are a labelling bug;
shared folds are the point.

Usage:
    python src/discover_pdb_binders.py --want 14
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"

# The PTM-clean subset of the panel. 1I8H is excluded: its peptide needs
# phosphothreonine (Pin1 WW reads pSer/pThr-Pro), so the canonical sequence
# the benchmark folds is not a binder at all.
EXISTING = ["1YCR", "1CKA", "1BE9", "1SEM", "1ELW", "2GBQ", "1D4T", "1TP5",
            "2FNT", "1NLO", "1OAI", "4LN2", "9F6S", "8KDX", "6YOO", "3DS4",
            "9GRF", "7S7J"]

# Receptors may share a fold; peptides may not share a sequence. The peptide
# threshold is deliberately strict -- substring containment (a histone tail
# truncation) already scores well above it.
RECEPTOR_MAX_ID = 0.90
PEPTIDE_MAX_ID = 0.60


# Purification tags and linkers crystallise in peptide-binding grooves and are
# returned by the structural screen looking exactly like ligands. A His-tag is
# not a binder, and scoring one as a true positive would corrupt the cognate
# class. 4W8H reached the accepted list with peptide "HHHHHH" before this.
TAG_MOTIFS = ("HHHHHH", "GSGSGS", "GGGGSG", "DYKDDDDK", "EQKLISEEDL",
              "WSHPQFEK", "YPYDVPDYA")


def is_tag_like(pep):
    """Purification tag, linker, or otherwise not a biological peptide."""
    if any(m in pep for m in TAG_MOTIFS):
        return True
    if len(set(pep)) < 4:                     # e.g. HHHHHH, AAAAAA, GSGSGS
        return True
    if max(pep.count(c) for c in set(pep)) / len(pep) > 0.6:
        return True
    return False


def cofactors(pdb_id, cache):
    """Components covalently attached to the peptide chain.

    The question is not whether the crystal contains a ligand -- most do -- but
    whether the *peptide* carries something the benchmark cannot fold. A metal
    sitting on the receptor or at a lattice contact changes nothing about
    whether the canonical peptide binds; a sugar bonded to its own serine does.

    Testing bound-ligand names alone flagged four of the 22-receptor panel
    (1ELW Ni, 6YOO Zn, 9GRF A2G, 7S7J Ca). Only one survives this test: 9GRF
    carries nine covale records onto chain B at Ser3 OG and Thr4 OG1, matching
    peptide AASTTTPAPA. StcE is a mucin-selective protease that reads the
    O-glycan, so the canonical peptide is not a binder -- the same defect that
    excluded 1I8H. The other three links are absent entirely.

    Returns [] on a network failure: a missed cofactor costs one imperfect panel
    member, whereas failing closed would silently shrink the panel whenever RCSB
    is slow.
    """
    cf = cache / f"{pdb_id}.covale.json"
    if cf.exists():
        return json.loads(cf.read_text())
    path = cache / f"{pdb_id}.cif"
    if not path.exists():
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "60",
             f"https://files.rcsb.org/download/{pdb_id}.cif", "-o", str(path)],
            capture_output=True, text=True)
        if proc.returncode != 0 or not path.exists() or path.stat().st_size == 0:
            return []
    try:
        import gemmi
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        # the peptide is the shortest polymer chain, matching fetch_chains()
        polys = [(len(ch.get_polymer()), ch.name) for ch in st[0]
                 if len(ch.get_polymer()) > 0]
        if len(polys) < 2:
            return []
        pep_chain = min(polys)[1]
        doc = gemmi.cif.read(str(path))
        blk = doc.sole_block()
        rows = blk.find("_struct_conn.", ["conn_type_id", "ptnr1_auth_asym_id",
                                          "ptnr1_label_comp_id",
                                          "ptnr2_auth_asym_id",
                                          "ptnr2_label_comp_id"])
        std = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
               "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
               "TYR", "VAL", "MSE"}
        found = []
        for r in rows:
            if r[0] != "covale":
                continue
            for me, them in ((1, 3), (3, 1)):
                if r[me] == pep_chain and r[them + 1] not in std:
                    found.append(r[them + 1])
        out = sorted(set(found))
        cf.write_text(json.dumps(out))
        return out
    except Exception:
        return []


def identity(a, b):
    """Similarity in [0,1]; substring containment scores near 1."""
    from difflib import SequenceMatcher
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def similar_to_any(seq, pool, threshold):
    return any(identity(seq, s) >= threshold for s in pool)


def search(rows=600, released_after=None):
    """Entry IDs matching the structural screen, best-resolution first."""
    query = {
        "query": {
            "type": "group", "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                                "operator": "equals", "value": 2}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.selected_polymer_entity_types",
                                "operator": "exact_match", "value": "Protein (only)"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                                "operator": "less_or_equal", "value": 2.5}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                                "operator": "range",
                                "value": {"from": 60, "to": 190}}},
            ] + ([] if released_after is None else [
                # Structures released after a model's training cutoff are the
                # only ones on which it can be said to *predict* rather than
                # recall. Boltz-1 and AlphaFold3 both cut off at 2021-09-30.
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_accession_info.initial_release_date",
                                "operator": "greater",
                                "value": released_after}},
            ]),
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                      "direction": "asc"}],
            "results_content_type": ["experimental"],
        },
    }
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "60", "-X", "POST", SEARCH_URL,
         "-H", "Content-Type: application/json", "-d", json.dumps(query)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"RCSB search failed: {proc.stderr[:200]}")
    data = json.loads(proc.stdout)
    return [r["identifier"] for r in data.get("result_set", [])]


def peptide_ptms(pdb_id):
    """Critical modified residues in the shortest polymer entity, or None.

    RCSB's FASTA gives canonical sequence -- phosphoserine reads as S,
    acetyl-lysine as K. A domain that recognises the modification does not bind
    the canonical peptide, so folding it makes the cognate pair a non-binder.
    entity_poly.pdbx_seq_one_letter_code keeps the modifications in parens.
    """
    from audit_panel_ptms import CRITICAL, audit
    a = audit(pdb_id, REPO_ROOT / "artifacts" / "ptm_audit")
    if not a or len(a["entities"]) < 2:
        return None
    return [m for m in a["entities"][0]["mods"] if m in CRITICAL]


def fetch_chains(pdb_id):
    """[(length, sequence, name)] sorted short-first, or None if unusable."""
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    text = subprocess.run(["curl", "-s", "--max-time", "30", url],
                          capture_output=True, text=True).stdout.strip()
    lines = [ln for ln in text.split("\n") if ln]
    if not lines or not lines[0].startswith(">"):
        return None
    chains = []
    for i in range(0, len(lines) - 1, 2):
        parts = lines[i].split("|")
        chains.append((len(lines[i + 1]), lines[i + 1],
                       parts[2][:40] if len(parts) > 2 else "?"))
    chains.sort()
    return chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", type=int, default=14,
                    help="how many NEW receptors to accept")
    ap.add_argument("--max-check", type=int, default=400)
    ap.add_argument("--receptor-max-id", type=float, default=RECEPTOR_MAX_ID,
                    help="reject a candidate whose receptor is at least this "
                         "identical to one already in the panel. The 0.90 "
                         "default admitted a 0.86 pair (8T33/8X8A, both GABA-A "
                         "receptor); a second panel wanting cleaner separation "
                         "should pass 0.80")
    ap.add_argument("--exclude", default="",
                    help="comma-separated PDB ids already in the panel being "
                         "built; excluded by id and deduplicated against")
    ap.add_argument("--released-after", default=None,
                    help="ISO date; keep only entries released after it, so the "
                         "panel is genuinely held out from the model's training set")
    ap.add_argument("--cache", default=str(REPO_ROOT / "artifacts" / "pdb_discovery"))
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "pdb_candidates.json"))
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    print("querying RCSB ...", flush=True)
    if args.released_after:
        print(f"  restricted to entries released after {args.released_after}")
    ids = search(released_after=args.released_after)
    print(f"  {len(ids)} entries match the structural screen")

    # EXISTING is the main panel. A caller building a *second* panel must pass
    # its members via --exclude, or they are neither skipped by id nor available
    # to deduplicate against, and near-duplicates of them get accepted.
    extra = [i.strip().upper() for i in args.exclude.split(",") if i.strip()]
    known = list(dict.fromkeys(EXISTING + extra))
    if extra:
        print(f"  + {len(extra)} caller-supplied receptors to exclude")
    ids = [i for i in ids if i.upper() not in known]

    # Existing receptors AND peptides, so additions are genuinely new on both
    # sides. Peptides matter most: a near-duplicate peptide turns another
    # receptor's decoy into a real binder.
    seen_receptors, seen_peptides = [], []
    # Every directory that may hold a panel receptor's sequence. Omitting one
    # silently disables deduplication against the receptors it contains:
    # pdb_binders_b2_n22 was missing here, so 9GRF (StcE) and 8KDX (Fyn SH3)
    # could not be matched, and their near-identical partners 9GRJ (0.98) and
    # 9GHK (0.96) were accepted into a later panel. Two receptors that similar
    # make each other's cognate peptide a mislabelled decoy.
    seqdirs = [REPO_ROOT / "artifacts" / d / "sequences"
               for d in ("pdb_binders", "pdb_binders_b2_n22", "pdb_binders_b2_n25",
                         "pdb_binders_b2", "heldout_panel")]
    for pid in known:
        for sd in seqdirs:
            f = sd / f"{pid}.json"
            if f.exists():
                d = json.loads(f.read_text())
                seen_receptors.append(d["receptor"])
                seen_peptides.append(d["peptide"])
                break
        else:
            print(f"  ! no cached sequence for {pid}; it cannot be deduplicated "
                  f"against", file=sys.stderr)
    print(f"  {len(seen_receptors)} existing receptors / "
          f"{len(seen_peptides)} peptides loaded")

    accepted, checked = [], 0
    for pid in ids:
        if len(accepted) >= args.want or checked >= args.max_check:
            break
        checked += 1
        cf = cache / f"{pid}.json"
        if cf.exists():
            chains = json.loads(cf.read_text())
        else:
            chains = fetch_chains(pid)
            if chains is not None:
                cf.write_text(json.dumps(chains))
        if not chains or len(chains) != 2:
            continue
        (plen, pseq, pname), (rlen, rseq, rname) = chains[0], chains[1]
        if not (6 <= plen <= 25 and 40 <= rlen <= 140):
            continue
        if not set(pseq) <= set("ACDEFGHIKLMNPQRSTVWY"):
            continue
        if not set(rseq) <= set("ACDEFGHIKLMNPQRSTVWY"):
            continue
        if is_tag_like(pseq):
            print(f"  - {pid} rejected: peptide {pseq} looks like a tag/linker, "
                  f"not a biological binder")
            continue
        if similar_to_any(rseq, seen_receptors, args.receptor_max_id):
            print(f"  - {pid} rejected: receptor {rlen}aa ({rname[:32]}) "
                  f">= {RECEPTOR_MAX_ID:.0%} identical to one already in the panel")
            continue
        if similar_to_any(pseq, seen_peptides, PEPTIDE_MAX_ID):
            print(f"  - {pid} rejected: peptide {pseq} too close to one already "
                  f"accepted (would make it a decoy that actually binds)")
            continue
        ptms = peptide_ptms(pid)
        if ptms:
            print(f"  - {pid} rejected: peptide needs {','.join(dict.fromkeys(ptms))}; "
                  f"canonical sequence is not binding-competent")
            continue
        cof = cofactors(pid, cache)
        if cof:
            print(f"  - {pid} rejected: interaction needs {', '.join(cof)}; "
                  f"the benchmark folds protein only")
            continue
        seen_receptors.append(rseq)
        seen_peptides.append(pseq)
        accepted.append({"pdb_id": pid, "receptor_len": rlen, "peptide_len": plen,
                         "peptide": pseq, "receptor_name": rname,
                         "peptide_name": pname})
        print(f"  + {pid}  receptor {rlen}aa ({rname[:32]}) | "
              f"peptide {plen}aa {pseq}", flush=True)

    Path(args.out).write_text(json.dumps(accepted, indent=2))
    print(f"\nchecked {checked} entries, accepted {len(accepted)}")
    print(f"wrote {args.out}")
    if len(accepted) < args.want:
        print(f"WARNING: wanted {args.want}, got {len(accepted)}", file=sys.stderr)
    print("\nPDB_IDS = " + json.dumps(known + [a["pdb_id"] for a in accepted]))


if __name__ == "__main__":
    main()
