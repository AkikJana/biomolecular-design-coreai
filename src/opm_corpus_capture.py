"""Capture OuterProductMean activations across a corpus of folds.

The single-target fit left the question open: held-out error was worse than train
(0.35 -> 0.43), which is what you expect when factors are fitted to one protein.
Settling whether the low-rank OPM can replace the stock layer on pretrained
weights needs many targets.

Two changes from opm_capture_activations.py:

* **Keying by module identity.** The earlier version bucketed by insertion order
  and stopped appending once the confidence module's OPM layers pushed the key
  count past the limit, so only one sample per layer was ever recorded.
* **Inputs only.** The stock output is ~8 MB per sample and is exactly
  recomputable from (m_norm, mask) and the layer weights, which are stored once
  per layer. Storing inputs alone is ~1 MB per sample, so a corpus fits in RAM.

Usage:
    python src/opm_corpus_capture.py --inputs <dir of boltz yamls> --max-per-layer 40
"""

import argparse
import pickle
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(REPO_ROOT / "artifacts/pdb_binders/batch_00/inputs"))
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "opm_corpus.pkl"))
    ap.add_argument("--work-dir", default=str(REPO_ROOT / "artifacts" / "opm_corpus_run"))
    ap.add_argument("--layers", type=int, default=4, help="distinct OPM layers to track")
    ap.add_argument("--max-per-layer", type=int, default=40)
    ap.add_argument("--sampling-steps", type=int, default=5)
    ap.add_argument("--recycling-steps", type=int, default=0)
    args = ap.parse_args()

    import boltz.model.layers.outer_product_mean as opm_mod

    samples, weights, allowed = {}, {}, {}
    original = opm_mod.OuterProductMeanStock.forward

    def patched(self, m, mask, chunk_size=None):
        key = allowed.get(id(self))
        if key is None and len(allowed) < args.layers:
            key = f"layer_{len(allowed)}"
            allowed[id(self)] = key
            with torch.no_grad():
                weights[key] = {
                    "proj_a": self.proj_a.weight.detach().float().cpu(),
                    "proj_b": self.proj_b.weight.detach().float().cpu(),
                    "proj_o_w": self.proj_o.weight.detach().float().cpu(),
                    "proj_o_b": self.proj_o.bias.detach().float().cpu(),
                }
        if key is not None:
            bucket = samples.setdefault(key, [])
            if len(bucket) < args.max_per_layer:
                with torch.no_grad():
                    bucket.append({"m_norm": self.norm(m).detach().float().cpu(),
                                   "mask": mask.detach().float().cpu()})
        return original(self, m, mask, chunk_size)

    opm_mod.OuterProductMeanStock.forward = patched
    n_yaml = len(list(Path(args.inputs).glob("*.yaml")))
    print(f"folding {n_yaml} inputs from {args.inputs}", flush=True)

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
    except Exception as exc:
        print(f"  fold ended early ({type(exc).__name__}: {str(exc)[:120]})")

    total = sum(len(v) for v in samples.values())
    print(f"\ncaptured {total} samples across {len(samples)} layers")
    for k, v in samples.items():
        print(f"  {k}: {len(v)} samples, m_norm {tuple(v[0]['m_norm'].shape)}")
    if not total:
        raise SystemExit("nothing captured")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"samples": samples, "weights": weights}, fh)
    size_mb = Path(args.out).stat().st_size / 1024**2
    print(f"saved: {args.out}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
