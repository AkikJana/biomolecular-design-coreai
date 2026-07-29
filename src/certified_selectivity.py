"""Risk-controlled cross-context binder selection.

This module implements the methodological core of a prospective-design study:
it certifies *robust selectivity*, rather than ranking a single predicted
affinity.  For a binder ``x`` it considers the weakest desired context and the
strongest counter-screen,

    S(x) = min(on-target activity) - max(off-target activity).

Joint split-conformal calibration estimates one residual radius for both terms.
The resulting lower bound is conservative: a candidate is certified only when
its worst-case desired activity remains above its worst-case off-target activity
by a user-declared margin.  The guarantee is marginal and relies on the usual
exchangeability assumption between calibration and future target-disjoint
examples; it is not a per-candidate biological guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from prospective_campaign import ContextualEvaluation


@dataclass(frozen=True)
class ContextualAssayRecord:
    """Observed robust-context assay outcomes for one previously evaluated binder."""

    evaluation: ContextualEvaluation
    experimental_on_target_score: float
    experimental_off_target_score: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.experimental_on_target_score) or not math.isfinite(self.experimental_off_target_score):
            raise ValueError("Experimental context scores must be finite.")

    @property
    def target_id(self) -> str:
        return self.evaluation.candidate.target_id

    @property
    def experimental_selectivity(self) -> float:
        return self.experimental_on_target_score - self.experimental_off_target_score


@dataclass(frozen=True)
class CertifiedSelectivity:
    """Jointly calibrated bounds for robust on-vs-off target selectivity."""

    evaluation: ContextualEvaluation
    predicted_selectivity: float
    lower_bound: float
    upper_bound: float
    residual_radius: float
    alpha: float

    def passes(self, minimum_selectivity: float) -> bool:
        return self.evaluation.verification.accepted and self.lower_bound >= minimum_selectivity


class JointSelectivityConformalCalibrator:
    """Split-conformal simultaneous interval for robust on/off context scores.

    A calibration residual is ``max(|on_obs-on_pred|, |off_obs-off_pred|)``.
    Thus, when the joint interval covers both quantities, the robust-selectivity
    interval has radius ``2 * residual_radius``.  Calibrating the pair together
    is the key distinction from independently calibrated affinity heads.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between zero and one.")
        self.alpha = alpha
        self.residuals: Tuple[float, ...] = ()
        self.train_targets: Tuple[str, ...] = ()

    @staticmethod
    def _quantile(residuals: Sequence[float], alpha: float) -> float:
        if not residuals:
            raise ValueError("At least one calibration record is required.")
        ordered = sorted(residuals)
        index = min(math.ceil((len(ordered) + 1) * (1.0 - alpha)) - 1, len(ordered) - 1)
        return ordered[index]

    def fit(
        self,
        calibration_records: Sequence[ContextualAssayRecord],
        train_target_ids: Iterable[str] = (),
    ) -> "JointSelectivityConformalCalibrator":
        train_targets = set(train_target_ids)
        calibration_targets = {record.target_id for record in calibration_records}
        if not calibration_targets:
            raise ValueError("At least one calibration target is required.")
        if train_targets & calibration_targets:
            raise ValueError("Calibration targets must be disjoint from model-training targets.")
        residuals = []
        for record in calibration_records:
            residual = max(
                abs(record.experimental_on_target_score - record.evaluation.on_target_score),
                abs(record.experimental_off_target_score - record.evaluation.off_target_score),
            )
            residuals.append(residual)
        self.residuals = tuple(residuals)
        self.train_targets = tuple(sorted(train_targets))
        return self

    def certify(self, evaluation: ContextualEvaluation, alpha: float | None = None) -> CertifiedSelectivity:
        if not self.residuals:
            raise RuntimeError("Call fit before certify.")
        alpha = self.alpha if alpha is None else alpha
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between zero and one.")
        radius = self._quantile(self.residuals, alpha)
        predicted = evaluation.specificity_score
        return CertifiedSelectivity(
            evaluation=evaluation,
            predicted_selectivity=predicted,
            lower_bound=predicted - 2.0 * radius,
            upper_bound=predicted + 2.0 * radius,
            residual_radius=radius,
            alpha=alpha,
        )


@dataclass(frozen=True)
class RiskControlledBatch:
    """A batch whose certification family-wise error is budgeted conservatively."""

    selected: Tuple[CertifiedSelectivity, ...]
    rejected: Tuple[CertifiedSelectivity, ...]
    minimum_selectivity: float
    familywise_error_budget: float
    per_candidate_alpha: float


def select_with_familywise_risk_control(
    evaluations: Sequence[ContextualEvaluation],
    calibrator: JointSelectivityConformalCalibrator,
    minimum_selectivity: float,
    familywise_error_budget: float = 0.1,
    max_candidates: int | None = None,
) -> RiskControlledBatch:
    """Select certified binders with a Bonferroni campaign-level error budget.

    Each decision receives ``budget / number_of_tested_candidates`` conformal
    error.  Under exchangeability plus a union bound, the probability of one or
    more false certifications is at most the requested budget.  This is
    intentionally conservative and makes the campaign claim easy to audit.
    """
    if not evaluations:
        raise ValueError("At least one evaluation is required.")
    if not 0.0 < familywise_error_budget < 1.0:
        raise ValueError("familywise_error_budget must be between zero and one.")
    if max_candidates is not None and max_candidates < 1:
        raise ValueError("max_candidates must be positive when supplied.")
    per_candidate_alpha = familywise_error_budget / len(evaluations)
    certified = [calibrator.certify(evaluation, alpha=per_candidate_alpha) for evaluation in evaluations]
    passing = [item for item in certified if item.passes(minimum_selectivity)]
    passing.sort(key=lambda item: item.lower_bound, reverse=True)
    if max_candidates is not None:
        selected = passing[:max_candidates]
    else:
        selected = passing
    selected_ids = {id(item) for item in selected}
    rejected = [item for item in certified if id(item) not in selected_ids]
    return RiskControlledBatch(
        selected=tuple(selected), rejected=tuple(rejected), minimum_selectivity=minimum_selectivity,
        familywise_error_budget=familywise_error_budget, per_candidate_alpha=per_candidate_alpha,
    )


@dataclass(frozen=True)
class ReferenceAllocation:
    """A candidate chosen for costly reference evaluation and its priority evidence."""

    certification: CertifiedSelectivity
    expected_information_gain: float
    priority_per_cost: float


class CostAwareReferenceAllocator:
    """Allocate a finite reference-model budget to uncertain decision boundaries.

    ``reference_residual_radius`` is an externally measured validation quantity
    for Boltz/reference evaluation.  The policy prioritizes valid candidates
    near the certification boundary when reference scoring is expected to shrink
    their interval, using a transparent value-of-information proxy rather than
    pretending an unrun model call has known outcome.
    """

    def __init__(self, reference_cost: float, reference_residual_radius: float):
        if reference_cost <= 0 or reference_residual_radius < 0:
            raise ValueError("reference_cost must be positive and residual radius non-negative.")
        self.reference_cost = reference_cost
        self.reference_residual_radius = reference_residual_radius

    def allocate(
        self,
        certifications: Sequence[CertifiedSelectivity],
        minimum_selectivity: float,
        budget: float,
    ) -> List[ReferenceAllocation]:
        if budget < 0:
            raise ValueError("budget must be non-negative.")
        capacity = int(budget // self.reference_cost)
        allocations = []
        for certification in certifications:
            if not certification.evaluation.verification.accepted:
                continue
            current_width = certification.upper_bound - certification.lower_bound
            reference_width = 4.0 * self.reference_residual_radius
            information_gain = max(0.0, current_width - reference_width)
            # Candidates nearest the threshold are most likely to change action
            # after a narrower reference interval.  The exponential is bounded
            # and deterministic, avoiding a fabricated posterior probability.
            scale = max(current_width / 2.0, 1e-8)
            boundary_weight = math.exp(-0.5 * ((certification.predicted_selectivity - minimum_selectivity) / scale) ** 2)
            allocations.append(ReferenceAllocation(
                certification, information_gain, boundary_weight * information_gain / self.reference_cost
            ))
        return sorted(allocations, key=lambda item: item.priority_per_cost, reverse=True)[:capacity]
