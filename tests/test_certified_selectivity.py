import pytest

from campaign_ranking import CandidateRecord
from certified_selectivity import (
    ContextualAssayRecord,
    CostAwareReferenceAllocator,
    JointSelectivityConformalCalibrator,
    select_with_familywise_risk_control,
)
from prospective_campaign import ContextualEvaluation, VerificationTrace


def _evaluation(target_id, on_target, off_target, accepted=True):
    specificity = on_target - off_target
    candidate = CandidateRecord(
        target_id, "TARGET", "DWDWDWDW", features={"specificity_score": specificity}, model_score=specificity
    )
    trace = VerificationTrace(accepted, {"all": accepted}, {}, () if accepted else ("hotspots",))
    return ContextualEvaluation(candidate, (), on_target, off_target, specificity, trace)


def _calibrator():
    calibration = [
        ContextualAssayRecord(_evaluation("cal-a", 0.8, 0.2), 0.85, 0.15),
        ContextualAssayRecord(_evaluation("cal-b", 0.7, 0.3), 0.65, 0.35),
        ContextualAssayRecord(_evaluation("cal-c", 0.9, 0.1), 0.95, 0.05),
    ]
    return JointSelectivityConformalCalibrator(alpha=0.2).fit(calibration, train_target_ids=("train-a",))


def test_joint_conformal_bounds_cover_robust_selectivity_and_enforce_target_disjointness():
    calibrator = _calibrator()
    certified = calibrator.certify(_evaluation("test", 0.9, 0.2))

    assert certified.predicted_selectivity == pytest.approx(0.7)
    assert certified.residual_radius == pytest.approx(0.05)
    assert certified.lower_bound == pytest.approx(0.6)
    assert certified.upper_bound == pytest.approx(0.8)
    assert certified.passes(0.55)

    with pytest.raises(ValueError, match="disjoint"):
        JointSelectivityConformalCalibrator().fit(
            [ContextualAssayRecord(_evaluation("train-a", 0.8, 0.2), 0.8, 0.2)],
            train_target_ids=("train-a",),
        )


def test_familywise_selection_and_cost_aware_allocation_are_conservative():
    calibrator = _calibrator()
    high = _evaluation("test-high", 0.9, 0.2)  # LCB = 0.6
    boundary = _evaluation("test-boundary", 0.65, 0.2)  # LCB = 0.35
    invalid = _evaluation("test-invalid", 0.95, 0.1, accepted=False)
    batch = select_with_familywise_risk_control(
        [high, boundary, invalid], calibrator, minimum_selectivity=0.5,
        familywise_error_budget=0.12,
    )

    assert batch.per_candidate_alpha == pytest.approx(0.04)
    assert [item.evaluation.candidate.target_id for item in batch.selected] == ["test-high"]
    assert len(batch.rejected) == 2

    certifications = [calibrator.certify(evaluation) for evaluation in (high, boundary, invalid)]
    allocation = CostAwareReferenceAllocator(reference_cost=2.0, reference_residual_radius=0.01).allocate(
        certifications, minimum_selectivity=0.5, budget=2.0
    )
    assert len(allocation) == 1
    assert allocation[0].certification.evaluation.candidate.target_id == "test-boundary"
