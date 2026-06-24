"""Boltz-2 predict_fn skeletons for the benchmark's reference scorer.

A predict_fn maps (target_seq, binder_seq) -> a dict with Boltz output fields
(complex_plddt, iptm, ptm, affinity_pred_value, affinity_probability_binary,
sample_atom_coords). The benchmark's reference scorer ranks binders from these.

Two practical paths:

1. Out-of-process (recommended, no featurization reimplementation):
   run `boltz predict` (which handles MSA/tokenization/atom features), then read
   the per-prediction confidence_*.json and affinity_*.json it writes.
   -> BoltzCliPredictFn / read_boltz_outputs.

2. In-process (full control, you supply featurization):
   load Boltz2 and call forward on feats built from the (target, binder) complex.
   -> make_inprocess_predict_fn (skeleton; featurize_fn is yours to provide,
   e.g. via boltz's BoltzInferenceDataModule).
"""

import json
import os
from typing import Callable, Dict, Optional

import torch

from benchmark_surrogate_vs_reference import Scorer

# Confidence fields the benchmark consumes (subset of confidence_*.json).
_CONF_KEYS = ("complex_plddt", "iptm", "ptm", "ligand_iptm", "protein_iptm", "confidence_score")


def read_boltz_outputs(results_dir: str, name: str, model_idx: int = 0) -> Dict[str, torch.Tensor]:
    """Read a single Boltz prediction's confidence (+ affinity, if present).

    Expects the layout written by `boltz predict`:
        {results_dir}/predictions/{name}/confidence_{name}_model_{i}.json
        {results_dir}/predictions/{name}/affinity_{name}.json            (Boltz-2 affinity)
    """
    pred_dir = os.path.join(results_dir, "predictions", name)
    conf_path = os.path.join(pred_dir, f"confidence_{name}_model_{model_idx}.json")
    with open(conf_path) as f:
        conf = json.load(f)

    out = {k: torch.tensor([float(conf[k])]) for k in _CONF_KEYS if k in conf}

    aff_path = os.path.join(pred_dir, f"affinity_{name}.json")
    if os.path.exists(aff_path):
        with open(aff_path) as f:
            aff = json.load(f)
        for k in ("affinity_pred_value", "affinity_probability_binary"):
            if k in aff:
                out[k] = torch.tensor([float(aff[k])])
    return out


class BoltzCliPredictFn:
    """predict_fn that reads pre-computed `boltz predict` outputs.

    name_fn maps (target, binder) -> the prediction name used on disk (the input
    record/YAML stem). Run boltz once over all pairs, then point this at the dir.
    """

    def __init__(self, results_dir: str, name_fn: Callable[[str, str], str], model_idx: int = 0):
        self.results_dir = results_dir
        self.name_fn = name_fn
        self.model_idx = model_idx

    def __call__(self, target: str, binder: str) -> Dict[str, torch.Tensor]:
        return read_boltz_outputs(self.results_dir, self.name_fn(target, binder), self.model_idx)


def make_inprocess_predict_fn(model, featurize_fn: Callable[[str, str], dict],
                              recycling_steps: int = 3, num_sampling_steps: int = 200,
                              device: str = "cpu") -> Callable[[str, str], Dict[str, torch.Tensor]]:
    """In-process Boltz-2 predict_fn (skeleton).

    model: a loaded Boltz2 with affinity_prediction=True.
    featurize_fn(target, binder) -> feats dict (the part you must supply, e.g. via
    boltz's BoltzInferenceDataModule / the same featurization `boltz predict` uses).
    """
    model.eval().to(device)

    @torch.no_grad()
    def predict_fn(target: str, binder: str) -> Dict[str, torch.Tensor]:
        feats = featurize_fn(target, binder)
        feats = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feats.items()}
        out = model(
            feats,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            diffusion_samples=1,
        )
        result = {k: out[k] for k in _CONF_KEYS if k in out}
        for k in ("affinity_pred_value", "affinity_probability_binary", "sample_atom_coords"):
            if k in out:
                result[k] = out[k]
        return result

    return predict_fn


class BoltzAffinityScorer(Scorer):
    """Reference scorer ranking binders by a Boltz-2 affinity field."""

    name = "boltz2"

    def __init__(self, predict_fn: Callable[[str, str], Dict[str, torch.Tensor]],
                 rank_key: str = "affinity_probability_binary",
                 higher_is_better: bool = True, size_bytes: Optional[int] = None):
        self.predict_fn = predict_fn
        self.rank_key = rank_key
        self.sign = 1.0 if higher_is_better else -1.0
        self._size = size_bytes

    @torch.no_grad()
    def score(self, pairs):
        vals = []
        for target, binder in pairs:
            out = self.predict_fn(target, binder)
            vals.append(self.sign * out[self.rank_key].reshape(-1)[0])
        return torch.stack(vals)

    def model_size_bytes(self):
        return self._size
