import torch

from agentic_design_loop import run_codesign_loop
from campaign_ranking import (
    CalibratedEnsembleRanker,
    CandidateRecord,
    TargetedCampaignReward,
    make_target_disjoint_split,
    partition_records,
)


def _labeled_records():
    records = []
    for target_index in range(8):
        for candidate_index in range(12):
            activity = candidate_index / 11.0
            penalty = 0.3 if candidate_index % 5 == 0 else 0.0
            records.append(
                CandidateRecord(
                    target_id=f"target-{target_index}",
                    target_sequence="MATEVLADIGSAKLR",
                    binder_sequence="A" * 8 + "W" * (candidate_index % 3),
                    features={"model_activity": activity, "developability_penalty": penalty},
                    experimental_score=2.5 * activity - penalty,
                )
            )
    return records


def test_target_disjoint_ranker_calibrates_and_enriches_top_candidates():
    records = _labeled_records()
    split = make_target_disjoint_split([record.target_id for record in records], seed=7)
    train, calibration, test = partition_records(records, split)

    assert not ({record.target_id for record in train} & {record.target_id for record in calibration})
    assert not ({record.target_id for record in train} & {record.target_id for record in test})
    assert not ({record.target_id for record in calibration} & {record.target_id for record in test})

    ranker = CalibratedEnsembleRanker(members=8, seed=4).fit(train, calibration)
    metrics = ranker.evaluate(test, top_k=4)
    acquisitions = ranker.select(test, k=4, exploration_weight=0.5, developability_weight=1.0)

    assert ranker.conformal_radius is not None
    assert metrics["selected_mean_experimental_score"] > metrics["overall_mean_experimental_score"]
    assert len(acquisitions) == 4
    assert all(item.prediction.lower <= item.prediction.mean <= item.prediction.upper for item in acquisitions)


def test_predictor_backed_reward_records_real_outputs_for_codesign():
    def predict_fn(target, binder):
        del target
        score = binder.count("W") / max(1, len(binder))
        coords = torch.arange(len(binder), dtype=torch.float32).view(1, -1, 1)
        coords = coords * torch.tensor([3.8, 0.0, 0.0])
        return {
            "complex_plddt": torch.tensor([0.5 + 0.4 * score]),
            "iptm": torch.tensor([0.3 + 0.6 * score]),
            "ptm": torch.tensor([0.3 + 0.6 * score]),
            "affinity_probability_binary": torch.tensor([score]),
            "sample_atom_coords": coords,
        }

    reward_model = TargetedCampaignReward(
        target_id="toy-target", target_sequence="MATEVLADIGSAKLR", predict_fn=predict_fn
    )
    history = run_codesign_loop(
        reward_model=reward_model,
        wt_sequence="MATEVLADIGSAKLR",
        interface_positions=[2, 4, 8, 12],
        iterations=1,
        group_size=4,
        inner_steps=1,
        device="cpu",
        verbose=False,
    )

    assert len(reward_model.ledger) == 4
    assert len(history[0]["candidate_records"]) == 4
    assert all(record.model_score is not None for record in reward_model.ledger)
    assert all("boltz_confidence" in record.features for record in reward_model.ledger)
