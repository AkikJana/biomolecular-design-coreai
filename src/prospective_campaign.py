"""Verifiable, multi-context prospective binder campaign utilities.

The module intentionally has no dependency on a particular structure predictor.
Callers provide ``predict_fn(target_sequence, binder_sequence)`` (for example a
Boltz adapter) and get an auditable decision record suitable for a prospective
campaign.  It complements :mod:`campaign_ranking` with four practical controls:

* robust aggregation across intended and off-target contexts;
* explicit sequence/interface constraints and machine-readable verifier traces;
* fixed-budget, cluster-aware campaign metrics; and
* target-conditional conformal routing between a cheap edge ranker and a costly
  reference predictor.

These utilities are a protocol implementation, not a claim of benchmark
performance.  Report real results only after running the same budget and assay
definitions on a declared target-disjoint benchmark or prospective campaign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from campaign_ranking import (
    CalibratedEnsembleRanker,
    CandidateRecord,
    PredictionFn,
    RankingPrediction,
    evaluate_candidates,
    sequence_developability_features,
)


@dataclass(frozen=True)
class TargetContext:
    """A structural/biological context used to assess one candidate.

    ``role`` is ``"on_target"`` for a desired state and ``"off_target"`` for a
    counter-screen.  Multiple desired states make the aggregate robust to
    conformational variation; counter-screens turn selectivity into an explicit
    optimization objective.
    """

    context_id: str
    target_sequence: str
    role: str = "on_target"

    def __post_init__(self) -> None:
        if not self.context_id or not self.target_sequence:
            raise ValueError("A context requires non-empty id and target_sequence.")
        if self.role not in {"on_target", "off_target"}:
            raise ValueError("context role must be 'on_target' or 'off_target'.")


@dataclass(frozen=True)
class DesignSpec:
    """Hard design requirements, including intended and counter-screen contexts."""

    target_id: str
    contexts: Tuple[TargetContext, ...]
    min_binder_length: int = 1
    max_binder_length: int = 256
    max_developability_penalty: float = 0.25
    min_hotspot_contact_fraction: float = 0.0
    required_hotspots: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must not be empty.")
        if not self.contexts or not any(context.role == "on_target" for context in self.contexts):
            raise ValueError("DesignSpec requires at least one on_target context.")
        if self.min_binder_length < 1 or self.max_binder_length < self.min_binder_length:
            raise ValueError("Binder length bounds are invalid.")
        if self.max_developability_penalty < 0 or not math.isfinite(self.max_developability_penalty):
            raise ValueError("max_developability_penalty must be finite and non-negative.")
        if not 0.0 <= self.min_hotspot_contact_fraction <= 1.0:
            raise ValueError("min_hotspot_contact_fraction must be between zero and one.")
        if len(set(self.required_hotspots)) != len(self.required_hotspots) or any(index < 0 for index in self.required_hotspots):
            raise ValueError("required_hotspots must be unique non-negative indices.")


@dataclass(frozen=True)
class VerificationTrace:
    """Audit trail for hard constraints; failed checks are never silently ignored."""

    accepted: bool
    checks: Mapping[str, bool]
    values: Mapping[str, float]
    reasons: Tuple[str, ...]


def verify_design(
    binder_sequence: str,
    spec: DesignSpec,
    evidence: Optional[Mapping[str, object]] = None,
) -> VerificationTrace:
    """Apply deterministic sequence and optional interface-evidence constraints.

    Predictors may supply ``hotspot_contact_fraction`` directly, or
    ``contacted_hotspots`` as a tuple/list of target indices.  If a positive
    hotspot threshold is requested but neither evidence form is supplied, the
    candidate fails closed rather than being treated as verified.
    """
    if not binder_sequence:
        raise ValueError("binder_sequence must not be empty.")
    evidence = evidence or {}
    features = sequence_developability_features(binder_sequence)
    length = len(binder_sequence)
    checks: Dict[str, bool] = {
        "length": spec.min_binder_length <= length <= spec.max_binder_length,
        "developability": features["developability_penalty"] <= spec.max_developability_penalty,
    }
    values: Dict[str, float] = {
        "binder_length": float(length),
        "developability_penalty": features["developability_penalty"],
    }

    hotspot_fraction: Optional[float] = None
    if "hotspot_contact_fraction" in evidence:
        hotspot_fraction = float(evidence["hotspot_contact_fraction"])
    elif spec.required_hotspots and "contacted_hotspots" in evidence:
        contacted = set(evidence["contacted_hotspots"])  # type: ignore[arg-type]
        hotspot_fraction = len(contacted & set(spec.required_hotspots)) / len(spec.required_hotspots)
    if hotspot_fraction is not None:
        if not math.isfinite(hotspot_fraction) or not 0.0 <= hotspot_fraction <= 1.0:
            raise ValueError("hotspot_contact_fraction must be finite and between zero and one.")
        values["hotspot_contact_fraction"] = hotspot_fraction
    hotspot_required = spec.min_hotspot_contact_fraction > 0.0 or bool(spec.required_hotspots)
    checks["hotspots"] = (not hotspot_required) or (
        hotspot_fraction is not None and hotspot_fraction >= spec.min_hotspot_contact_fraction
    )

    reasons = tuple(name for name, passed in checks.items() if not passed)
    return VerificationTrace(not reasons, checks, values, reasons)


@dataclass(frozen=True)
class ContextualEvaluation:
    """All predictor outputs collapsed into a robust, specificity-aware record."""

    candidate: CandidateRecord
    context_records: Tuple[CandidateRecord, ...]
    on_target_score: float
    off_target_score: float
    specificity_score: float
    verification: VerificationTrace


def evaluate_across_contexts(
    binder_sequence: str,
    spec: DesignSpec,
    predict_fn: PredictionFn,
    interface_evidence: Optional[Mapping[str, object]] = None,
    off_target_weight: float = 1.0,
) -> ContextualEvaluation:
    """Score a binder across contexts using worst-case desired activity.

    Desired contexts are aggregated by their minimum score, while counter-screen
    activity is their maximum score.  This prevents a high score in one pose from
    masking a failed state or a promiscuous off-target interaction.
    """
    if off_target_weight < 0 or not math.isfinite(off_target_weight):
        raise ValueError("off_target_weight must be finite and non-negative.")
    raw = [
        CandidateRecord(
            target_id=f"{spec.target_id}:{context.context_id}",
            target_sequence=context.target_sequence,
            binder_sequence=binder_sequence,
        )
        for context in spec.contexts
    ]
    scored = evaluate_candidates(raw, predict_fn)
    on_target = [record.model_score for record, context in zip(scored, spec.contexts) if context.role == "on_target"]
    off_target = [record.model_score for record, context in zip(scored, spec.contexts) if context.role == "off_target"]
    assert all(score is not None for score in on_target + off_target)
    on_target_score = min(float(score) for score in on_target)
    off_target_score = max((float(score) for score in off_target), default=0.0)
    specificity_score = on_target_score - off_target_weight * off_target_score
    verification = verify_design(binder_sequence, spec, interface_evidence)
    features = dict(sequence_developability_features(binder_sequence))
    features.update({
        "on_target_worst_score": on_target_score,
        "off_target_best_score": off_target_score,
        "specificity_score": specificity_score,
        "verified": float(verification.accepted),
    })
    candidate = CandidateRecord(
        target_id=spec.target_id,
        target_sequence=next(context.target_sequence for context in spec.contexts if context.role == "on_target"),
        binder_sequence=binder_sequence,
        features=features,
        model_score=specificity_score,
    )
    return ContextualEvaluation(candidate, tuple(scored), on_target_score, off_target_score, specificity_score, verification)


@dataclass(frozen=True)
class FixedBudgetReport:
    candidate_count: int
    elapsed_seconds: float
    projected_candidates_per_24h: float
    top_k: int
    top_k_success_rate: float
    top_k_cluster_count: int
    top_k_mean_score: float


def fixed_budget_report(
    candidates: Sequence[CandidateRecord],
    elapsed_seconds: float,
    success_threshold: float,
    top_k: int = 10,
    cluster_fn: Optional[Callable[[CandidateRecord], str]] = None,
) -> FixedBudgetReport:
    """Compute a compact fixed-budget report compatible with prospective studies.

    Experimental scores are used when present; otherwise this deliberately
    reports predictor-level success.  The caller should label the latter as
    *in-silico* rather than assay success.
    """
    if not candidates:
        raise ValueError("At least one candidate is required.")
    if elapsed_seconds <= 0 or not math.isfinite(elapsed_seconds):
        raise ValueError("elapsed_seconds must be finite and positive.")
    if top_k < 1:
        raise ValueError("top_k must be at least one.")
    scores = []
    for candidate in candidates:
        score = candidate.experimental_score if candidate.experimental_score is not None else candidate.model_score
        if score is None or not math.isfinite(float(score)):
            raise ValueError("Every candidate needs a finite experimental_score or model_score.")
        scores.append(float(score))
    chosen = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)[: min(top_k, len(candidates))]
    clusters = {cluster_fn(candidate) if cluster_fn else candidate.binder_sequence for candidate, _ in chosen}
    return FixedBudgetReport(
        candidate_count=len(candidates),
        elapsed_seconds=elapsed_seconds,
        projected_candidates_per_24h=len(candidates) * 86_400.0 / elapsed_seconds,
        top_k=len(chosen),
        top_k_success_rate=sum(score >= success_threshold for _, score in chosen) / len(chosen),
        top_k_cluster_count=len(clusters),
        top_k_mean_score=sum(score for _, score in chosen) / len(chosen),
    )


@dataclass(frozen=True)
class RoutingDecision:
    candidate: CandidateRecord
    prediction: Optional[RankingPrediction]
    route: str
    reason: str


class TargetConditionalConformalRouter:
    """Escalate uncertain candidates from an edge ranker to a reference model.

    Calibration residuals are stratified by target when enough calibration
    samples exist.  New targets safely fall back to the global target-disjoint
    conformal radius rather than borrowing an unvalidated target-specific one.
    """

    def __init__(
        self,
        ranker: CalibratedEnsembleRanker,
        min_lower_bound: float,
        max_interval_width: float,
        min_target_calibration_samples: int = 3,
    ) -> None:
        if max_interval_width < 0 or min_target_calibration_samples < 1:
            raise ValueError("Routing thresholds must be non-negative and sample count positive.")
        self.ranker = ranker
        self.min_lower_bound = min_lower_bound
        self.max_interval_width = max_interval_width
        self.min_target_calibration_samples = min_target_calibration_samples
        self.target_radii: Dict[str, float] = {}

    def fit(self, calibration_records: Sequence[CandidateRecord]) -> "TargetConditionalConformalRouter":
        if self.ranker.conformal_radius is None:
            raise RuntimeError("Fit the ranker before fitting its router.")
        grouped: Dict[str, list[float]] = {}
        for record, prediction in zip(calibration_records, self.ranker.predict(calibration_records)):
            if record.experimental_score is None:
                raise ValueError("Router calibration requires experimental_score for every record.")
            grouped.setdefault(record.target_id, []).append(abs(float(record.experimental_score) - prediction.mean))
        self.target_radii = {
            target: max(values)
            for target, values in grouped.items()
            if len(values) >= self.min_target_calibration_samples
        }
        return self

    def predict(self, candidate: CandidateRecord) -> RankingPrediction:
        prediction = self.ranker.predict([candidate])[0]
        radius = self.target_radii.get(candidate.target_id, self.ranker.conformal_radius)
        assert radius is not None
        return RankingPrediction(prediction.mean, prediction.std, prediction.mean - radius, prediction.mean + radius)

    def route(self, evaluation: ContextualEvaluation) -> RoutingDecision:
        if not evaluation.verification.accepted:
            return RoutingDecision(evaluation.candidate, None, "reject", "failed:" + ",".join(evaluation.verification.reasons))
        prediction = self.predict(evaluation.candidate)
        width = prediction.upper - prediction.lower
        if prediction.lower >= self.min_lower_bound and width <= self.max_interval_width:
            return RoutingDecision(evaluation.candidate, prediction, "edge", "confident_lower_bound")
        return RoutingDecision(evaluation.candidate, prediction, "reference", "uncertain_or_low_lower_bound")
