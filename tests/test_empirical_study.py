import pytest

from campaign_ranking import CandidateRecord
from certified_selectivity import ContextualAssayRecord, JointSelectivityConformalCalibrator
from empirical_study import build_prospective_assay_manifest, evaluate_target_disjoint_study
from prospective_campaign import ContextualEvaluation, VerificationTrace


def _record(target_id, binder, predicted_selectivity, observed_selectivity):
    candidate = CandidateRecord(target_id, "TARGET", binder, model_score=predicted_selectivity)
    evaluation = ContextualEvaluation(
        candidate=candidate,
        context_records=(),
        on_target_score=predicted_selectivity + 0.1,
        off_target_score=0.1,
        specificity_score=predicted_selectivity,
        verification=VerificationTrace(True, {"length": True, "hotspots": True}, {}, ()),
    )
    return ContextualAssayRecord(
        evaluation, experimental_on_target_score=observed_selectivity + 0.1, experimental_off_target_score=0.1
    )


def _records():
    records = []
    for target in range(10):
        records.extend([
            _record(f"target-{target}", f"GOOD{target}", 0.8, 0.75),
            _record(f"target-{target}", f"BAD{target}", 0.2, 0.25),
        ])
    return records


def test_target_disjoint_study_reports_baselines_coverage_and_budget_replay():
    result = evaluate_target_disjoint_study(
        _records(), minimum_selectivity=0.5, seed=3, alphas=(0.1, 0.2),
        familywise_error_budget=0.2, fixed_budget_seconds=120.0, seconds_per_candidate=30.0,
    )

    assert set(result.split.train_targets).isdisjoint(result.split.calibration_targets)
    assert len(result.baselines) == 3
    assert result.baselines[0].policy == "raw_predicted_selectivity"
    assert result.baselines[2].precision == pytest.approx(1.0)
    assert [point.alpha for point in result.coverage_curve] == [0.1, 0.2]
    assert all(point.empirical_coverage == pytest.approx(1.0) for point in result.coverage_curve)
    assert result.fixed_budget[0].evaluated_count == 4
    assert result.fixed_budget[0].successful_discoveries == 2


def test_prospective_manifest_is_deduplicated_and_ranked_by_lower_bound():
    records = _records()
    calibrator = JointSelectivityConformalCalibrator().fit(records[:4], train_target_ids=("different-train-target",))
    high = calibrator.certify(records[4].evaluation)
    duplicate = calibrator.certify(records[4].evaluation)
    low = calibrator.certify(records[5].evaluation)
    manifest = build_prospective_assay_manifest([low, duplicate, high], minimum_selectivity=0.5, max_requests=2)

    assert len(manifest) == 1
    assert manifest[0].binder_sequence == records[4].evaluation.candidate.binder_sequence
    assert manifest[0].certified_lower_bound >= 0.5
