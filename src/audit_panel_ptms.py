"""Flag benchmark peptides whose binding depends on chemistry Boltz cannot see.

RCSB's FASTA endpoint returns the *canonical* sequence: modified residues are
collapsed to their parent amino acid. Phosphoserine becomes S, acetyl-lysine
becomes K, phosphotyrosine becomes Y. The benchmark folds those canonical
sequences, so for a complex whose interaction requires the modification, the
"cognate" pair is not binding-competent as folded -- an SH2 domain binds
phospho-tyrosine, not tyrosine, and a bromodomain reads acetyl-lysine, not
lysine.

That is a mislabelled positive, and it inflates the apparent failure of the
scoring signal: the benchmark asks ipTM to rank a pair that does not bind.

`entity_poly.pdbx_seq_one_letter_code` keeps modified residues in parentheses,
so the two representations can be compared directly. This audits every entry in
the panel and reports which peptides carry modifications.

Usage:
    python src/audit_panel_ptms.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pdb_binder_benchmark import PDB_IDS  # noqa: E402

DATA = "https://data.rcsb.org/rest/v1/core"

# Modified residues that carry the recognition chemistry for a whole class of
# peptide-binding domains. If one of these sits in the peptide, folding the
# canonical sequence removes the reason the complex exists.
CRITICAL = {
    "SEP": "phosphoserine", "TPO": "phosphothreonine", "PTR": "phosphotyrosine",
    "ALY": "acetyl-lysine", "MLZ": "methyl-lysine", "MLY": "dimethyl-lysine",
    "M3L": "trimethyl-lysine", "AGM": "methyl-arginine", "DA2": "methyl-arginine",
    "CIR": "citrulline", "HIC": "methyl-histidine", "SEC": "selenocysteine",
    "CSO": "oxidised cysteine", "NEP": "phosphohistidine",
}


def get(url):
    p = subprocess.run(["curl", "-s", "--max-time", "30", url],
                       capture_output=True, text=True)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def audit(pdb_id, cache_dir):
    cache = cache_dir / f"{pdb_id}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    entry = get(f"{DATA}/entry/{pdb_id}")
    if not entry:
        return None
    eids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
    entities = []
    for eid in eids:
        e = get(f"{DATA}/polymer_entity/{pdb_id}/{eid}")
        if not e:
            continue
        poly = e.get("entity_poly", {})
        raw = poly.get("pdbx_seq_one_letter_code", "") or ""
        can = poly.get("pdbx_seq_one_letter_code_can", "") or ""
        mods = re.findall(r"\(([A-Z0-9]{2,3})\)", raw)
        entities.append({"entity_id": eid, "length": len(can.replace("\n", "")),
                         "canonical": can.replace("\n", ""),
                         "raw": raw.replace("\n", ""), "mods": mods})
    entities.sort(key=lambda x: x["length"])
    out = {"pdb_id": pdb_id, "entities": entities}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2))
    return out


def main():
    cache_dir = REPO_ROOT / "artifacts" / "ptm_audit"
    flagged, clean, failed = [], [], []

    for pid in PDB_IDS:
        a = audit(pid, cache_dir)
        if not a or len(a["entities"]) < 2:
            failed.append(pid)
            continue
        pep = a["entities"][0]
        crit = [m for m in pep["mods"] if m in CRITICAL]
        other = [m for m in pep["mods"] if m not in CRITICAL]
        if crit:
            flagged.append((pid, pep, crit))
            names = ", ".join(f"{m} ({CRITICAL[m]})" for m in dict.fromkeys(crit))
            print(f"  FLAG {pid}  {pep['canonical']:<26} needs {names}")
        else:
            clean.append(pid)
            note = f"  (non-critical: {','.join(other)})" if other else ""
            print(f"  ok   {pid}  {pep['canonical']:<26}{note}")

    print(f"\n{len(clean)} clean, {len(flagged)} PTM-dependent, {len(failed)} unresolved")
    if failed:
        print(f"  unresolved: {failed}")
    print("\nclean panel:")
    print("PDB_IDS = " + json.dumps(clean))
    (REPO_ROOT / "artifacts" / "ptm_audit_result.json").write_text(json.dumps(
        {"clean": clean, "flagged": [f[0] for f in flagged],
         "flagged_detail": [{"pdb_id": f[0], "peptide": f[1]["canonical"],
                             "raw": f[1]["raw"], "critical_mods": f[2]}
                            for f in flagged],
         "unresolved": failed}, indent=2))


if __name__ == "__main__":
    main()
