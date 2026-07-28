"""Reproducible target-disjoint evaluation for certified-selectivity design.

This module analyses assay-labelled :class:`ContextualAssayRecord` objects.  It
does not manufacture biological results: a study report is only as valid as its
assay scores, target split, context definitions, and pre-declared thresholds.
The utilities make those choices explicit and make baseline comparisons
repeatable before a prospective campaign is launched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from campaign_ranking import TargetDisjointSplit, make_target_disjoint_split
from certified_selectivity import (
    CertifiedSelectivity,
    ContextualAssayRecord,
    JointSelectivityConformalCalibrator,
    select_with_familywise_risk_control,
)


@dataclass(frozen=True)
class SelectionMetrics:
    policy: str
    selected_count: int
    selection_rate: float
    precision: float
    recall: float
    mean_experimental_selectivity: float
    cluster_count: int


@dataclass(frozen=True)
class CoveragePoint:
    alpha: float
    empirical_coverage: float
    mean_interval_width: float
    certified_count: int
    certified_precision: float


@dataclass(frozen=True)
class FixedBudgetReplay:
    policy: str
    budget_seconds: float
    used_seconds: float
    evaluated_count: int
    successful_discoveries: int
    discovery_rate_per_24h: float
    cluster_count: int


@dataclass(frozen=True)
class EmpiricalStudyResult:
    split: TargetDisjointSplit
    baselines: Tuple[SelectionMetrics, ...]
    coverage_curve: Tuple[CoveragePoint, ...]
    fixed_budget: Tuple[FixedBudgetReplay, ...]


def _selection_metrics(
    policy: str,
    selected: Sequence[ContextualAssayRecord],
    all_records: Sequence[ContextualAssayRecord],
    minimum_selectivity: float,
    cluster_fn: Callable[[ContextualAssayRecord], str],
) -> SelectionMetrics:
    positives = [record for record in all_records if record.experimental_selectivity >= minimum_selectivity]
    true_selected = [record for record in selected if record.experimental_selectivity >= minimum_selectivity]
    precision = len(true_selected) / len(selected) if selected else 0.0
    recall = len(true_selected) / len(positives) if positives else 0.0
    return SelectionMetrics(
        policy=policy,
        selected_count=len(selected),
        selection_rate=len(selected) / len(all_records),
        precision=precision,
        recall=recall,
        mean_experimental_selectivity=(
            sum(record.experimental_selectivity for record in selected) / len(selected) if selected else 0.0
        ),
        cluster_count=len({cluster_fn(record) for record in selected}),
    )


def fixed_budget_replay(
    policy: str,
    ranked_records: Sequence[ContextualAssayRecord],
    budget_seconds: float,
    seconds_per_candidate: float,
    minimum_selectivity: float,
    cluster_fn: Optional[Callable[[ContextualAssayRecord], str]] = None,
) -> FixedBudgetReplay:
    """Replay a pre-ranked policy under a fixed compute/assay-equivalent budget."""
    if budget_seconds <= 0 or seconds_per_candidate <= 0:
        raise ValueError("Budget and per-candidate time must be positive.")
    capacity = min(len(ranked_records), int(budget_seconds // seconds_per_candidate))
    evaluated = list(ranked_records[:capacity])
    successes = [record for record in evaluated if record.experimental_selectivity >= minimum_selectivity]
    cluster_fn = cluster_fn or (lambda record: record.evaluation.candidate.binder_sequence)
    used = capacity * seconds_per_candidate
    return FixedBudgetReplay(
        policy=policy,
        budget_seconds=budget_seconds,
        used_seconds=used,
        evaluated_count=capacity,
        successful_discoveries=len(successes),
        discovery_rate_per_24h=(len(successes) * 86_400.0 / used) if used else 0.0,
        cluster_count=len({cluster_fn(record) for record in successes}),
    )


def evaluate_target_disjoint_study(
    records: Sequence[ContextualAssayRecord],
    minimum_selectivity: float,
    split: Optional[TargetDisjointSplit] = None,
    seed: int = 0,
    alphas: Sequence[float] = (0.05, 0.1, 0.2),
    familywise_error_budget: float = 0.1,
    fixed_budget_seconds: float = 86_400.0,
    seconds_per_candidate: float = 60.0,
    cluster_fn: Optional[Callable[[ContextualAssayRecord], str]] = None,
) -> EmpiricalStudyResult:
    """Compare raw ranking, verifier-gated ranking, and risk-controlled design.

    ``records`` must already contain target-context assay outcomes.  The train
    partition is reserved for fitting the upstream predictor; this function only
    calibrates on the calibration targets and reports all metrics on disjoint
    test targets.
    """
    if not records:
        raise ValueError("At least one assay-labelled record is required.")
    if not all(0.0 < alpha < 1.0 for alpha in alphas):
        raise ValueError("Every alpha must be between zero and one.")
    split = split or make_target_disjoint_split((record.target_id for record in records), seed=seed)
    partition: Dict[str, List[ContextualAssayRecord]] = {"train": [], "calibration": [], "test": []}
    target_to_partition = {
        **{target: "train" for target in split.train_targets},
        **{target: "calibration" for target in split.calibration_targets},
        **{target: "test" for target in split.test_targets},
    }
    for record in records:
        try:
            partition[target_to_partition[record.target_id]].append(record)
        except KeyError as error:
            raise ValueError(f"Target '{record.target_id}' is absent from the split.") from error
    if not partition["calibration"] or not partition["test"]:
        raise ValueError("Calibration and test partitions must both contain assay records.")
    cluster_fn = cluster_fn or (lambda record: record.evaluation.candidate.binder_sequence)
    calibrator = JointSelectivityConformalCalibrator(alpha=min(alphas)).fit(
        partition["calibration"], train_target_ids=split.train_targets
    )
    test = partition["test"]
    raw = sorted(test, key=lambda record: record.evaluation.specificity_score, reverse=True)
    verifier_gated = [record for record in raw if record.evaluation.verification.accepted]
    batch = select_with_familywise_risk_control(
        [record.evaluation for record in test], calibrator, minimum_selectivity, familywise_error_budget
    )
    lookup = {id(record.evaluation): record for record in test}
    certified = [lookup[id(item.evaluation)] for item in batch.selected]
    baselines = (
        _selection_metrics("raw_predicted_selectivity", [record for record in raw if record.evaluation.specificity_score >= minimum_selectivity], test, minimum_selectivity, cluster_fn),
        _selection_metrics("verifier_gated_selectivity", [record for record in verifier_gated if record.evaluation.specificity_score >= minimum_selectivity], test, minimum_selectivity, cluster_fn),
        _selection_metrics("joint_conformal_risk_controlled", certified, test, minimum_selectivity, cluster_fn),
    )
    curve = []
    for alpha in sorted(alphas):
        certifications = [calibrator.certify(record.evaluation, alpha=alpha) for record in test]
        covered = [
            item.lower_bound <= record.experimental_selectivity <= item.upper_bound
            for item, record in zip(certifications, test)
        ]
        selected = [
            record for item, record in zip(certifications, test) if item.passes(minimum_selectivity)
        ]
        successes = [record for record in selected if record.experimental_selectivity >= minimum_selectivity]
        curve.append(CoveragePoint(
            alpha=alpha,
            empirical_coverage=sum(covered) / len(covered),
            mean_interval_width=sum(item.upper_bound - item.lower_bound for item in certifications) / len(certifications),
            certified_count=len(selected),
            certified_precision=len(successes) / len(selected) if selected else 0.0,
        ))
    fixed = (
        fixed_budget_replay("raw_predicted_selectivity", raw, fixed_budget_seconds, seconds_per_candidate, minimum_selectivity, cluster_fn),
        fixed_budget_replay("verifier_gated_selectivity", verifier_gated, fixed_budget_seconds, seconds_per_candidate, minimum_selectivity, cluster_fn),
        fixed_budget_replay("joint_conformal_risk_controlled", certified, fixed_budget_seconds, seconds_per_candidate, minimum_selectivity, cluster_fn),
    )
    return EmpiricalStudyResult(split, baselines, tuple(curve), fixed)


@dataclass(frozen=True)
class ProspectiveAssayRequest:
    """A traceable laboratory request; does not perform or fabricate an assay."""

    target_id: str
    binder_sequence: str
    predicted_selectivity: float
    certified_lower_bound: float
    requested_contexts: Tuple[str, ...]
    verifier_checks: Tuple[str, ...]


def build_prospective_assay_manifest(
    certifications: Iterable[CertifiedSelectivity],
    minimum_selectivity: float,
    max_requests: int,
) -> Tuple[ProspectiveAssayRequest, ...]:
    """Build a deduplicated, lower-bound-ranked assay manifest from certificates."""
    if max_requests < 1:
        raise ValueError("max_requests must be at least one.")
    accepted = [item for item in certifications if item.passes(minimum_selectivity)]
    accepted.sort(key=lambda item: item.lower_bound, reverse=True)
    requests = []
    seen = set()
    for item in accepted:
        candidate = item.evaluation.candidate
        key = (candidate.target_id, candidate.binder_sequence)
        if key in seen:
            continue
        seen.add(key)
        requests.append(ProspectiveAssayRequest(
            target_id=candidate.target_id,
            binder_sequence=candidate.binder_sequence,
            predicted_selectivity=item.predicted_selectivity,
            certified_lower_bound=item.lower_bound,
            requested_contexts=tuple(record.target_id for record in item.evaluation.context_records),
            verifier_checks=tuple(name for name, passed in item.evaluation.verification.checks.items() if passed),
        ))
        if len(requests) == max_requests:
            break
    return tuple(requests)
