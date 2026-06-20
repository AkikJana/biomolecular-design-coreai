try:
    import pytest
except ImportError:
    pytest = None

import torch
from src.receptor_inr_compression import (
    ReceptorNeuralField,
    generate_mock_receptor,
    train_inr,
    evaluate_inr
)

def test_generate_mock_receptor():
    n_residues = 100
    d_embed = 64
    coords, targets = generate_mock_receptor(n_residues=n_residues, d_embed=d_embed, seed=42)
    
    assert coords.shape == (n_residues, 3)
    assert targets.shape == (n_residues, d_embed)
    assert coords.dtype == torch.float32
    assert targets.dtype == torch.float32
    # Verify coordinates are normalized between -1 and 1
    assert torch.all(coords >= -1.0) and torch.all(coords <= 1.0)


def test_receptor_neural_field_forward(mode):
    n_residues = 50
    d_embed = 128
    coords, _ = generate_mock_receptor(n_residues=n_residues, d_embed=d_embed)
    
    model = ReceptorNeuralField(
        in_features=3,
        hidden_features=32,
        hidden_layers=2,
        out_features=d_embed,
        mode=mode
    )
    
    preds = model(coords)
    assert preds.shape == (n_residues, d_embed)
    assert preds.dtype == torch.float32


def test_training_and_evaluation():
    n_residues = 100
    d_embed = 64
    coords, targets = generate_mock_receptor(n_residues=n_residues, d_embed=d_embed)
    
    model = ReceptorNeuralField(
        in_features=3,
        hidden_features=16,
        hidden_layers=1,
        out_features=d_embed,
        mode="siren"
    )
    
    device = torch.device("cpu")
    
    # Train for a few epochs
    trained_model, loss_history, train_time = train_inr(
        model=model,
        coords=coords,
        targets=targets,
        epochs=10,
        lr=1e-3,
        device=device
    )
    
    assert len(loss_history) == 10
    assert loss_history[-1] < loss_history[0]  # Overfitting should decrease loss
    assert train_time > 0
    
    # Evaluate
    metrics = evaluate_inr(trained_model, coords, targets, device)
    
    assert "mse" in metrics
    assert "cosine_similarity" in metrics
    assert "r2_score" in metrics
    assert "num_params" in metrics
    assert "compression_factor" in metrics
    assert metrics["cosine_similarity"] >= -1.0 and metrics["cosine_similarity"] <= 1.0
    assert metrics["num_params"] > 0
    assert metrics["compression_factor"] > 0

if __name__ == "__main__":
    print("Running tests manually...")
    test_generate_mock_receptor()
    print("test_generate_mock_receptor passed!")
    for mode in ["siren", "pe_mlp", "basic_mlp"]:
        test_receptor_neural_field_forward(mode)
        print(f"test_receptor_neural_field_forward({mode}) passed!")
    test_training_and_evaluation()
    print("test_training_and_evaluation passed!")
    print("All tests passed successfully!")
