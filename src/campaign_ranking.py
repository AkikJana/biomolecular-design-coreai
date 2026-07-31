"""Target-disjoint, uncertainty-aware binder ranking for prospective campaigns.

This module is deliberately predictor-agnostic: ``predict_fn(target, binder)``
may wrap Boltz-2, AlphaFold-derived features, a local surrogate, or a cached
result reader.  It turns those outputs into a stable feature schema, trains an
ensemble only on *other targets*, calibrates a conformal uncertainty interval on
held-out targets, and selects candidates using a utility that balances expected
activity, uncertainty, and developability.

Experimental scores use a higher-is-better convention (for example ``-log10(Kd)``
or an assay enrichment).  Model-derived scores are never treated as ground truth:
they are features and proposal rewards until replaced by an assay result.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from boltz_reward import RewardModel, boltz_confidence_score, clash_penalty, compute_design_reward


PredictionFn = Callable[[str, str], Dict[str, torch.Tensor]]


@dataclass(frozen=True)
class CandidateRecord:
    """One binder-target proposal and its available model or assay evidence."""

    target_id: str
    target_sequence: str
    binder_sequence: str
    features: Mapping[str, float] = field(default_factory=dict)
    model_score: Optional[float] = None
    experimental_score: Optional[float] = None


@dataclass(frozen=True)
class TargetDisjointSplit:
    """A split of target identifiers; no target may occur in two partitions."""

    train_targets: Tuple[str, ...]
    calibration_targets: Tuple[str, ...]
    test_targets: Tuple[str, ...]


@dataclass(frozen=True)
class RankingPrediction:
    mean: float
    std: float
    lower: float
    upper: float


@dataclass(frozen=True)
class Acquisition:
    candidate: CandidateRecord
    prediction: RankingPrediction
    utility: float


def _as_scalar(value: torch.Tensor, key: str) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"Prediction field '{key}' must be a scalar tensor.")
    result = float(value.detach().cpu().reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"Prediction field '{key}' must be finite.")
    return result


def sequence_developability_features(sequence: str) -> Dict[str, float]:
    """Small, transparent sequence-level developability feature set.

    These are screening features, not a replacement for expression, aggregation,
    immunogenicity, or biophysical assays.
    """
    if not sequence:
        raise ValueError("binder_sequence must not be empty.")
    length = len(sequence)
    hydrophobic = sum(residue in "AVILMFWY" for residue in sequence) / length
    cysteine = sequence.count("C") / length
    charge = (sum(residue in "KR" for residue in sequence) - sum(residue in "DE" for residue in sequence)) / length
    max_run = 1
    current_run = 1
    for previous, current in zip(sequence, sequence[1:]):
        current_run = current_run + 1 if current == previous else 1
        max_run = max(max_run, current_run)
    penalty = max(0.0, hydrophobic - 0.45) + max(0.0, cysteine - 0.08) + max(0.0, (max_run - 3) / length)
    return {
        "binder_length": float(length),
        "hydrophobic_fraction": hydrophobic,
        "cysteine_fraction": cysteine,
        "net_charge_fraction": charge,
        "max_homopolymer_fraction": max_run / length,
        "developability_penalty": penalty,
    }


def features_from_prediction(
    output: Mapping[str, torch.Tensor], binder_sequence: str, clash_weight: float = 1.0
) -> Tuple[Dict[str, float], float]:
    """Extract a stable ranking schema and a real predictor reward.

    Boltz affinity probability is preferred when present.  Otherwise the reward
    falls back to Boltz confidence minus the geometric clash penalty.  This keeps
    the design loop usable for structure-only predictors without fabricating a
    target label.
    """
    features = sequence_developability_features(binder_sequence)

    if "complex_plddt" in output:
        features["boltz_confidence"] = _as_scalar(boltz_confidence_score(dict(output)), "boltz_confidence")
    if "affinity_probability_binary" in output:
        features["affinity_probability"] = _as_scalar(output["affinity_probability_binary"], "affinity_probability_binary")
    if "affinity_pred_value" in output:
        features["affinity_value"] = _as_scalar(output["affinity_pred_value"], "affinity_pred_value")
    if "sample_atom_coords" in output:
        coords = output["sample_atom_coords"]
        mask = output.get("atom_mask")
        features["clash_penalty"] = _as_scalar(clash_penalty(coords, mask), "clash_penalty")

    if "affinity_probability_binary" in output:
        model_score = features["affinity_probability"]
    elif "complex_plddt" in output:
        model_score = _as_scalar(compute_design_reward(dict(output), clash_weight), "design_reward")
    else:
        raise ValueError(
            "Prediction output requires affinity_probability_binary or complex_plddt."
        )
    return features, model_score


def evaluate_candidates(
    candidates: Iterable[CandidateRecord], predict_fn: PredictionFn, clash_weight: float = 1.0
) -> List[CandidateRecord]:
    """Attach predictor-derived features and proposal rewards to candidates."""
    evaluated = []
    for candidate in candidates:
        output = predict_fn(candidate.target_sequence, candidate.binder_sequence)
        features, model_score = features_from_prediction(output, candidate.binder_sequence, clash_weight)
        evaluated.append(replace(candidate, features=features, model_score=model_score))
    return evaluated


class TargetedCampaignReward(RewardModel):
    """Adapt a pairwise predictor to the existing sequence-only GRPO loop.

    ``ledger`` records exactly which candidates and predictor features informed
    each update.  A real prospective campaign can later fill
    ``experimental_score`` for the same records and retrain the ranker.
    """

    def __init__(
        self,
        target_id: str,
        target_sequence: str,
        predict_fn: PredictionFn,
        clash_weight: float = 1.0,
    ):
        if not target_id or not target_sequence:
            raise ValueError("target_id and target_sequence must not be empty.")
        self.target_id = target_id
        self.target_sequence = target_sequence
        self.predict_fn = predict_fn
        self.clash_weight = clash_weight
        self.last_records: List[CandidateRecord] = []
        self.ledger: List[CandidateRecord] = []

    @torch.no_grad()
    def score(self, sequences: List[str]) -> torch.Tensor:
        candidates = [
            CandidateRecord(self.target_id, self.target_sequence, sequence)
            for sequence in sequences
        ]
        self.last_records = evaluate_candidates(candidates, self.predict_fn, self.clash_weight)
        self.ledger.extend(self.last_records)
        return torch.tensor([record.model_score for record in self.last_records], dtype=torch.float32)


def make_target_disjoint_split(
    target_ids: Iterable[str], seed: int = 0, train_fraction: float = 0.6, calibration_fraction: float = 0.2
) -> TargetDisjointSplit:
    """Create a deterministic train/calibration/test split at the target level."""
    unique = sorted(set(target_ids))
    if len(unique) < 3:
        raise ValueError("At least three distinct targets are required for a target-disjoint split.")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("train_fraction and calibration_fraction must be between 0 and 1.")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("train_fraction + calibration_fraction must leave a test partition.")

    rng = random.Random(seed)
    rng.shuffle(unique)
    num_targets = len(unique)
    num_train = max(1, round(num_targets * train_fraction))
    num_calibration = max(1, round(num_targets * calibration_fraction))
    if num_train + num_calibration >= num_targets:
        num_train, num_calibration = num_targets - 2, 1
    split = TargetDisjointSplit(
        train_targets=tuple(unique[:num_train]),
        calibration_targets=tuple(unique[num_train : num_train + num_calibration]),
        test_targets=tuple(unique[num_train + num_calibration :]),
    )
    assert_target_disjoint(split)
    return split


def assert_target_disjoint(split: TargetDisjointSplit) -> None:
    partitions = [set(split.train_targets), set(split.calibration_targets), set(split.test_targets)]
    if any(not partition for partition in partitions):
        raise ValueError("Every target-disjoint partition must contain at least one target.")
    if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
        raise ValueError("A target appears in more than one target-disjoint partition.")


def partition_records(
    records: Sequence[CandidateRecord], split: TargetDisjointSplit
) -> Tuple[List[CandidateRecord], List[CandidateRecord], List[CandidateRecord]]:
    """Partition records and reject targets not explicitly assigned to the split."""
    assert_target_disjoint(split)
    partitions = {"train": [], "calibration": [], "test": []}
    target_partition = {
        **{target: "train" for target in split.train_targets},
        **{target: "calibration" for target in split.calibration_targets},
        **{target: "test" for target in split.test_targets},
    }
    for record in records:
        if record.target_id not in target_partition:
            raise ValueError(f"Target '{record.target_id}' is absent from the split.")
        partitions[target_partition[record.target_id]].append(record)
    return partitions["train"], partitions["calibration"], partitions["test"]


class CalibratedEnsembleRanker:
    """Bootstrap ridge ensemble with target-disjoint conformal calibration."""

    def __init__(self, members: int = 16, ridge: float = 1e-3, alpha: float = 0.1, seed: int = 0):
        if members < 2:
            raise ValueError("members must be at least 2 to estimate uncertainty.")
        if ridge < 0:
            raise ValueError("ridge must be non-negative.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        self.members, self.ridge, self.alpha, self.seed = members, ridge, alpha, seed
        self.feature_names: Tuple[str, ...] = ()
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.conformal_radius: Optional[float] = None

    @staticmethod
    def _labels(records: Sequence[CandidateRecord]) -> torch.Tensor:
        labels = []
        for record in records:
            if record.experimental_score is None:
                raise ValueError("Ranker fitting and evaluation require experimental_score for every record.")
            label = float(record.experimental_score)
            if not math.isfinite(label):
                raise ValueError("experimental_score values must be finite.")
            labels.append(label)
        if not labels:
            raise ValueError("At least one labeled record is required.")
        return torch.tensor(labels, dtype=torch.float64)

    def _matrix(self, records: Sequence[CandidateRecord]) -> torch.Tensor:
        if not self.feature_names:
            raise RuntimeError("Ranker is not fitted.")
        rows = []
        for record in records:
            missing = set(self.feature_names) - set(record.features)
            if missing:
                raise ValueError(f"Candidate is missing required feature(s): {sorted(missing)}")
            row = [float(record.features[name]) for name in self.feature_names]
            if not all(math.isfinite(value) for value in row):
                raise ValueError("Candidate features must be finite.")
            rows.append(row)
        if not rows:
            raise ValueError("At least one candidate is required.")
        return torch.tensor(rows, dtype=torch.float64)

    @staticmethod
    def _infer_feature_names(records: Sequence[CandidateRecord]) -> Tuple[str, ...]:
        if not records:
            raise ValueError("At least one candidate is required.")
        shared = set(records[0].features)
        for record in records[1:]:
            shared &= set(record.features)
        if not shared:
            raise ValueError("Candidates have no shared numeric features.")
        return tuple(sorted(shared))

    def fit(
        self,
        train_records: Sequence[CandidateRecord],
        calibration_records: Sequence[CandidateRecord],
        feature_names: Optional[Sequence[str]] = None,
    ) -> "CalibratedEnsembleRanker":
        train_targets = {record.target_id for record in train_records}
        calibration_targets = {record.target_id for record in calibration_records}
        if not train_targets or not calibration_targets:
            raise ValueError("Train and calibration data must both contain at least one target.")
        if train_targets & calibration_targets:
            raise ValueError("Calibration targets must be disjoint from training targets.")

        self.feature_names = tuple(feature_names) if feature_names else self._infer_feature_names(train_records)
        if not self.feature_names:
            raise ValueError("At least one feature is required.")
        train_x, train_y = self._matrix(train_records), self._labels(train_records)
        self.feature_mean = train_x.mean(dim=0)
        self.feature_std = train_x.std(dim=0, unbiased=False).clamp_min(1e-8)
        normalized = (train_x - self.feature_mean) / self.feature_std
        design = torch.cat([torch.ones((normalized.shape[0], 1), dtype=normalized.dtype), normalized], dim=1)

        generator = torch.Generator().manual_seed(self.seed)
        penalty = torch.eye(design.shape[1], dtype=design.dtype) * self.ridge
        penalty[0, 0] = 0.0  # do not regularize the intercept
        members = []
        for _ in range(self.members):
            indices = torch.randint(0, design.shape[0], (design.shape[0],), generator=generator)
            x_bootstrap, y_bootstrap = design[indices], train_y[indices]
            system = x_bootstrap.T @ x_bootstrap + penalty
            rhs = x_bootstrap.T @ y_bootstrap
            members.append(torch.linalg.solve(system, rhs))
        self.weights = torch.stack(members)

        # Permit provisional ensemble means for calibration.  The zero radius is
        # immediately replaced by the target-disjoint conformal residual quantile.
        self.conformal_radius = 0.0
        calibration_y = self._labels(calibration_records)
        calibration_mean = torch.tensor([prediction.mean for prediction in self.predict(calibration_records)], dtype=torch.float64)
        residuals = torch.sort(torch.abs(calibration_y - calibration_mean)).values
        quantile_index = min(math.ceil((len(residuals) + 1) * (1.0 - self.alpha)) - 1, len(residuals) - 1)
        self.conformal_radius = float(residuals[quantile_index])
        return self

    def predict(self, candidates: Sequence[CandidateRecord]) -> List[RankingPrediction]:
        if self.weights is None or self.feature_mean is None or self.feature_std is None or self.conformal_radius is None:
            raise RuntimeError("Call fit before predict.")
        x = self._matrix(candidates)
        normalized = (x - self.feature_mean) / self.feature_std
        design = torch.cat([torch.ones((normalized.shape[0], 1), dtype=normalized.dtype), normalized], dim=1)
        member_predictions = design @ self.weights.T
        mean = member_predictions.mean(dim=1)
        std = member_predictions.std(dim=1, unbiased=False)
        return [
            RankingPrediction(
                mean=float(mu), std=float(sigma),
                lower=float(mu - self.conformal_radius), upper=float(mu + self.conformal_radius),
            )
            for mu, sigma in zip(mean, std)
        ]

    def select(
        self,
        candidates: Sequence[CandidateRecord],
        k: int,
        exploration_weight: float = 1.0,
        developability_weight: float = 1.0,
    ) -> List[Acquisition]:
        if k < 1:
            raise ValueError("k must be at least 1.")
        if exploration_weight < 0 or developability_weight < 0:
            raise ValueError("Acquisition weights must be non-negative.")
        predictions = self.predict(candidates)
        acquisitions = []
        for candidate, prediction in zip(candidates, predictions):
            penalty = float(candidate.features.get("developability_penalty", 0.0))
            utility = prediction.mean + exploration_weight * prediction.std - developability_weight * penalty
            acquisitions.append(Acquisition(candidate, prediction, utility))
        return sorted(acquisitions, key=lambda acquisition: acquisition.utility, reverse=True)[: min(k, len(acquisitions))]

    def evaluate(self, records: Sequence[CandidateRecord], top_k: int = 5) -> Dict[str, float]:
        predictions = self.predict(records)
        labels = self._labels(records)
        means = torch.tensor([prediction.mean for prediction in predictions], dtype=torch.float64)
        lowers = torch.tensor([prediction.lower for prediction in predictions], dtype=torch.float64)
        uppers = torch.tensor([prediction.upper for prediction in predictions], dtype=torch.float64)
        k = min(max(1, top_k), len(records))
        selected = torch.topk(means, k=k).indices
        return {
            "rmse": float(torch.sqrt(torch.mean((means - labels) ** 2))),
            "interval_coverage": float(((labels >= lowers) & (labels <= uppers)).double().mean()),
            "selected_mean_experimental_score": float(labels[selected].mean()),
            "overall_mean_experimental_score": float(labels.mean()),
        }
