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

**The peptide filter is the one that matters.** Decoys are built by handing a
receptor a peptide from a different entry, so if two receptors bind homologous
peptides that "decoy" is a genuine binder -- a mislabelled negative. A first
pass deduplicating only receptors accepted 5GJI and 5AUL (identical peptide
SDYMNMTP) and three separate histone H3 tail peptides (ARTKQTARKSTGGKA /
ARTKQTARKST / ARTKQTAAKA), any pair of which would have poisoned the decoy
class for the others.

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

EXISTING = ["1YCR", "1CKA", "1BE9", "1SEM", "1ELW", "2GBQ", "1D4T", "1TP5",
            "1I8H", "2FNT", "1NLO"]

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


def identity(a, b):
    """Similarity in [0,1]; substring containment scores near 1."""
    from difflib import SequenceMatcher
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def similar_to_any(seq, pool, threshold):
    return any(identity(seq, s) >= threshold for s in pool)


def search(rows=600):
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
            ],
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
    ap.add_argument("--cache", default=str(REPO_ROOT / "artifacts" / "pdb_discovery"))
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "pdb_candidates.json"))
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    print("querying RCSB ...", flush=True)
    ids = search()
    print(f"  {len(ids)} entries match the structural screen")
    ids = [i for i in ids if i.upper() not in EXISTING]

    # Existing receptors AND peptides, so additions are genuinely new on both
    # sides. Peptides matter most: a near-duplicate peptide turns another
    # receptor's decoy into a real binder.
    seen_receptors, seen_peptides = [], []
    seqdir = REPO_ROOT / "artifacts" / "pdb_binders" / "sequences"
    for pid in EXISTING:
        f = seqdir / f"{pid}.json"
        if f.exists():
            d = json.loads(f.read_text())
            seen_receptors.append(d["receptor"])
            seen_peptides.append(d["peptide"])
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
        if similar_to_any(rseq, seen_receptors, RECEPTOR_MAX_ID):
            print(f"  - {pid} rejected: receptor {rlen}aa ({rname[:32]}) "
                  f">= {RECEPTOR_MAX_ID:.0%} identical to one already in the panel")
            continue
        if similar_to_any(pseq, seen_peptides, PEPTIDE_MAX_ID):
            print(f"  - {pid} rejected: peptide {pseq} too close to one already "
                  f"accepted (would make it a decoy that actually binds)")
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
    print("\nPDB_IDS = " + json.dumps(EXISTING + [a["pdb_id"] for a in accepted]))


if __name__ == "__main__":
    main()
