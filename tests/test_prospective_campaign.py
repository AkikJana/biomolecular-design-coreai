import pytest
import torch

from campaign_ranking import CalibratedEnsembleRanker, CandidateRecord
from prospective_campaign import (
    DesignSpec,
    TargetConditionalConformalRouter,
    TargetContext,
    evaluate_across_contexts,
    fixed_budget_report,
)


def _predict(target, binder):
    score = binder.count("W") / len(binder)
    if target.startswith("off"):
        score *= 0.9
    elif target.startswith("weak"):
        score *= 0.6
    return {"affinity_probability_binary": torch.tensor(score)}


def _spec(target_id="target"):
    return DesignSpec(
        target_id=target_id,
        contexts=(
            TargetContext("state-a", "on-a"),
            TargetContext("state-b", "weak-on-b"),
            TargetContext("counter", "off-counter", "off_target"),
        ),
        min_binder_length=6,
        max_binder_length=12,
        max_developability_penalty=0.8,
        min_hotspot_contact_fraction=0.5,
        required_hotspots=(10, 12),
    )


def test_multicontext_scoring_is_robust_and_verifier_fails_closed():
    evaluation = evaluate_across_contexts(
        "AAWWWWAA", _spec(), _predict, interface_evidence={"contacted_hotspots": (10,)}
    )

    assert len(evaluation.context_records) == 3
    assert evaluation.on_target_score == pytest.approx(0.3)  # weak desired context is limiting
    assert evaluation.off_target_score == pytest.approx(0.45)  # counter-screen uses its strongest score
    assert evaluation.specificity_score == pytest.approx(-0.15)
    assert evaluation.verification.accepted
    assert evaluation.candidate.features["verified"] == 1.0

    failed = evaluate_across_contexts("AAAA", _spec(), _predict)
    assert not failed.verification.accepted
    assert set(failed.verification.reasons) == {"length", "hotspots"}


def test_fixed_budget_report_reports_selection_success_and_diversity():
    candidates = [
        CandidateRecord("t", "T", "AAAAAA", model_score=0.1),
        CandidateRecord("t", "T", "BBBBBB", model_score=0.9),
        CandidateRecord("t", "T", "CCCCCC", model_score=0.8),
    ]
    report = fixed_budget_report(
        candidates, elapsed_seconds=30.0, success_threshold=0.75, top_k=2,
        cluster_fn=lambda record: record.binder_sequence[0],
    )

    assert report.candidate_count == 3
    assert report.top_k_success_rate == 1.0
    assert report.top_k_cluster_count == 2
    assert report.projected_candidates_per_24h == 8640.0


def _ranker_and_router():
    train, calibration = [], []
    for target, output in [("train-a", train), ("train-b", train), ("cal-a", calibration)]:
        for count in range(1, 6):
            score = count / 5.0
            output.append(CandidateRecord(
                target, "T", "A" * (10 - count) + "W" * count,
                features={"specificity_score": score, "developability_penalty": 0.0},
                experimental_score=2.0 * score,
            ))
    ranker = CalibratedEnsembleRanker(members=6, seed=2).fit(
        train, calibration, feature_names=("specificity_score", "developability_penalty")
    )
    return TargetConditionalConformalRouter(ranker, min_lower_bound=0.5, max_interval_width=0.4).fit(calibration)


def _router_spec():
    return DesignSpec(
        target_id="cal-a",
        contexts=(TargetContext("state", "on-a"),),
        min_binder_length=6,
        max_binder_length=12,
        max_developability_penalty=0.8,
        min_hotspot_contact_fraction=0.5,
    )


def test_target_conditional_router_escalates_low_confidence_or_failed_designs():
    router = _ranker_and_router()
    high = evaluate_across_contexts(
        "DWDWDWDWDD", _router_spec(), _predict, interface_evidence={"hotspot_contact_fraction": 1.0}
    )
    low = evaluate_across_contexts(
        "DDDDDDDDDW", _router_spec(), _predict, interface_evidence={"hotspot_contact_fraction": 1.0}
    )
    rejected = evaluate_across_contexts("AAAA", _router_spec(), _predict)

    assert router.route(high).route == "edge"
    assert router.route(low).route == "reference"
    assert router.route(rejected).route == "reject"
