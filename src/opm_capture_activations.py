"""Capture real OuterProductMean inputs/outputs from a live Boltz forward pass.

The CP projection experiment measured the low-rank OPM's error on `torch.randn`
inputs, which is the worst case: a function can be far lower-rank on the manifold
its inputs actually occupy than its weight tensor suggests. Post-LayerNorm MSA
representations are strongly correlated across channels and nothing like white
noise.

This records (m, mask, stock_output) triples from an actual fold, so the
approximation error can be measured -- and the low-rank factors fitted -- on the
real input distribution rather than a synthetic one.

Works by monkeypatching OuterProductMeanStock.forward before invoking the Boltz
CLI in-process, so the whole featurisation pipeline (MSA, tokenisation, atom
features) is the real one.

Usage:
    python src/opm_capture_activations.py --max-per-layer 4
"""

import argparse
import pickle
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(REPO_ROOT / "artifacts/pdb_binders/batch_00/inputs"),
                    help="a directory of boltz YAML inputs to fold")
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "opm_activations.pkl"))
    ap.add_argument("--work-dir", default=str(REPO_ROOT / "artifacts" / "opm_capture"))
    ap.add_argument("--max-per-layer", type=int, default=4,
                    help="samples retained per OPM layer (they are large)")
    ap.add_argument("--layers", type=int, default=4, help="how many distinct layers to keep")
    ap.add_argument("--sampling-steps", type=int, default=5)
    ap.add_argument("--recycling-steps", type=int, default=0)
    args = ap.parse_args()

    import boltz.model.layers.outer_product_mean as opm_mod

    captured = {}
    original = opm_mod.OuterProductMeanStock.forward

    def patched(self, m, mask, chunk_size=None):
        out = original(self, m, mask, chunk_size)
        key = getattr(self, "_capture_key", None)
        if key is None:
            key = f"layer_{len(captured)}"
            self._capture_key = key
        bucket = captured.setdefault(key, [])
        if len(bucket) < args.max_per_layer and len(captured) <= args.layers:
            # Store the *normalised* input, since that is what the bilinear form
            # consumes, alongside the exact stock output as the fitting target.
            with torch.no_grad():
                bucket.append({
                    "m_norm": self.norm(m).detach().float().cpu(),
                    "mask": mask.detach().float().cpu(),
                    "out": out.detach().float().cpu(),
                    "proj_a": self.proj_a.weight.detach().float().cpu(),
                    "proj_b": self.proj_b.weight.detach().float().cpu(),
                    "proj_o_w": self.proj_o.weight.detach().float().cpu(),
                    "proj_o_b": self.proj_o.bias.detach().float().cpu(),
                })
        return out

    opm_mod.OuterProductMeanStock.forward = patched
    print(f"patched {opm_mod.OuterProductMeanStock.__name__}.forward; folding...", flush=True)

    from boltz.main import cli
    argv = ["predict", args.inputs, "--out_dir", args.work_dir, "--model", "boltz1",
            "--accelerator", "cpu", "--recycling_steps", str(args.recycling_steps),
            "--sampling_steps", str(args.sampling_steps), "--output_format", "pdb",
            "--override", "--subsample_msa", "--num_subsampled_msa", "32",
            "--max_msa_seqs", "32"]
    try:
        cli.main(args=argv, standalone_mode=False)
    except SystemExit:
        pass
    except Exception as exc:                      # folding may abort after capture
        print(f"  fold ended early ({type(exc).__name__}: {str(exc)[:120]})")

    kept = {k: v for k, v in captured.items() if v}
    total = sum(len(v) for v in kept.values())
    print(f"\ncaptured {total} samples across {len(kept)} layers")
    for k, v in kept.items():
        s = v[0]
        print(f"  {k}: {len(v)} samples, m_norm {tuple(s['m_norm'].shape)}, "
              f"out {tuple(s['out'].shape)}")
    if not total:
        raise SystemExit("no activations captured -- is the stock OPM in use?")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(kept, fh)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
