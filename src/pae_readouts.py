"""PAE-derived interface readouts, computed identically for every panel.

Section 7.6 tested six readouts and found interface pLDDT the only one that
separates a peptide from its own scramble. The four readouts here were either
absent from that comparison or computed by a script that no longer exists in the
repository -- artifacts/pae_metrics.json has no producer -- so they are
reimplemented once, here, and used for both panels. A metric comparison across
panels is only meaningful if the metric is the same function on both.

  iface_pae     mean interchain PAE over interface residue pairs
  mpae          smallest interchain PAE anywhere: the "minimum PAE" that Shen
                et al. report as the best enrichment metric for AF3 and Boltz-2
                on DEKOIS2.0, and claim is insensitive to the target
  pae_frac_lt10 fraction of interchain pairs confidently placed
  ipsae         Dunbrack's interface pTM restricted to residues under a PAE
                cutoff, with d0 recomputed from the count that survives it
  pdockq2       pDockQ with its contact-count term replaced by a PAE term

pDockQ2 matters specifically. Section 7.6 showed pDockQ fails on short peptides
because scrambled peptides make *more* inter-chain contacts than cognates
(38.5 against 32.9), so its contact term cancels the interface-pLDDT signal it
multiplies. pDockQ2 replaces exactly that term. If the explanation in 7.6 is
right, pDockQ2 should recover -- which makes it a test of the mechanism, not
just another metric.

Lower is better for iface_pae and mpae; the callers flip their sign.
"""

import numpy as np

PAE_CUTOFF = 15.0        # ipSAE default
CONTACT_A = 8.0          # CB-CB, CA for glycine
D0_PDOCKQ2 = 10.0


def _coords(residues):
    return np.array([r["CB"].coord if "CB" in r else r["CA"].coord
                     for r in residues])


def _chains(model):
    ch = [[r for r in c if "CA" in r] for c in model]
    ch = [c for c in ch if c]
    return ch if len(ch) == 2 else None


def _d0(n):
    """pTM's length normalisation; the floor keeps tiny interfaces finite."""
    n = max(n, 19)
    return 1.24 * (n - 15) ** (1 / 3) - 1.8


def readouts(model, pae):
    """All PAE readouts for one two-chain model, or None if it is unusable.

    `pae` is the full residue-by-residue matrix in the model's residue order,
    which is chain A then chain B as boltz writes it.
    """
    chains = _chains(model)
    if chains is None:
        return None
    A, B = chains
    na, nb = len(A), len(B)
    if pae.shape[0] < na + nb:
        return None
    xa, xb = _coords(A), _coords(B)
    dist = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
    ia, ib = np.where(dist < CONTACT_A)
    if len(ia) == 0:
        return None

    # interchain blocks, both directions
    ab = pae[:na, na:na + nb]
    ba = pae[na:na + nb, :na]
    inter = np.concatenate([ab.ravel(), ba.ravel()])

    out = {
        "iface_pae": float(np.mean([ab[i, j] for i, j in zip(ia, ib)])),
        "mpae": float(inter.min()),
        "pae_frac_lt10": float((inter < 10.0).mean()),
    }

    # ipSAE: per-residue interface pTM over partners under the cutoff, with d0
    # from the surviving count rather than the whole chain -- that recomputation
    # is the point of the metric.
    best = 0.0
    for block in (ab, ba):
        for i in range(block.shape[0]):
            keep = block[i][block[i] < PAE_CUTOFF]
            if keep.size == 0:
                continue
            d0 = _d0(keep.size)
            best = max(best, float(np.mean(1.0 / (1.0 + (keep / d0) ** 2))))
    out["ipsae"] = best

    # pDockQ2
    plddt = np.array([r["CA"].bfactor for r in A] + [r["CA"].bfactor for r in B])
    idx_a, idx_b = np.unique(ia), np.unique(ib)
    iface_plddt = plddt[np.concatenate([idx_a, idx_b + na])].mean()
    sub = pae[np.ix_(idx_a, idx_b + na)]
    pae_term = float(np.mean(1.0 / (1.0 + (sub / D0_PDOCKQ2) ** 2)))
    x = iface_plddt * pae_term
    out["pdockq2"] = float(1.31 / (1.0 + np.exp(-0.075 * (x - 84.733))) + 0.005)
    return out


def load_pae(pred_dir, name):
    """The full PAE matrix boltz writes under --write_full_pae, or None."""
    f = pred_dir / f"pae_{name}_model_0.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z[z.files[0]]
