import os
import pytest
import torch
import torch.nn as nn
import numpy as np
import math
import time

# Import paths (repo root, src, boltz/src) come from
# [tool.pytest.ini_options] pythonpath in pyproject.toml.
from low_rank_pair_representation import LowRankTensorProduct, LowRankPairUpdater, FullRankPairUpdater
from cfg_distillation import TeacherVectorField, CFGDistilledVectorField, SinusoidalEmbedding, initialize_student_from_teacher
from speculative_flow_matching import SpeculativeFlowMatchingSampler
from train_neural_refiner import ResNetCoordinateRefiner, compute_supervised_loss, generate_mock_ground_truth
from quantized_attention_weights import DynamicQuantizedLinear
from bidirectional_design import BidirectionalCoDesigner, JointBiophysicalLoss
from boltz.model.layers.attention import AttentionPairBias
from boltz.model.modules.utils import autocast_device_type

try:
    from predict_structure import DynamicStructurePredictor
    HAS_COREAI = True
except ImportError:
    HAS_COREAI = False

# Loading an AOT CoreAI asset enters a native runtime and can terminate the
# Python process when the installed runtime, hardware, or model artifact is
# incompatible.  Keep that hardware integration opt-in; this suite's default
# contract is portable functional coverage through LightweightPredictor.
RUN_COREAI_INTEGRATION = os.environ.get("RUN_COREAI_INTEGRATION") == "1"

# Helper: Kabsch alignment and RMSD calculation
def calculate_rmsd(coords1, coords2):
    if isinstance(coords1, np.ndarray):
        coords1 = torch.from_numpy(coords1)
    if isinstance(coords2, np.ndarray):
        coords2 = torch.from_numpy(coords2)
    if coords1.ndim == 3:
        coords1 = coords1[0]
    if coords2.ndim == 3:
        coords2 = coords2[0]
    
    # Center coordinates
    c1 = coords1 - coords1.mean(dim=0)
    c2 = coords2 - coords2.mean(dim=0)
    
    # Compute covariance matrix
    covariance = torch.matmul(c1.t(), c2)
    U, S, V = torch.svd(covariance)
    R = torch.matmul(U, V.t())
    
    # Correct for reflection
    if torch.det(R) < 0:
        V = V.clone()
        V[:, -1] *= -1
        R = torch.matmul(U, V.t())
        
    c1_aligned = torch.matmul(c1, R)
    rmsd = torch.sqrt(torch.mean(torch.sum((c1_aligned - c2) ** 2, dim=-1)))
    return rmsd.item()

# Helper: activation retention
def saved_element_count(build_output):
    """Count tensor elements retained for backward by `build_output()`.

    This is the quantity the low-rank pair factorization exists to reduce:
    full-rank materializes an O(N^2 * d_mid) intermediate, low-rank keeps only
    the O(N * rank) factors. Measuring it beats asserting a precomputed constant.
    Returns (element_count, output).
    """
    total = 0

    def pack(tensor):
        nonlocal total
        total += tensor.numel()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = build_output()
    return total, output


# Helper: backbone continuity
def backbone_continuity_cv(coords) -> float:
    """Coefficient of variation of consecutive CA-CA distances.

    A predicted chain has near-constant spacing between consecutive residues, so
    this is small; an unstructured point cloud has no such regularity. It is used
    in place of an absolute RMSD bound in the Tier 4 tests because it actually
    separates model output from noise -- measured over 40 random initialisations,
    the predictor stays under 0.19 while `torch.randn` never drops below 0.33.
    """
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)
    if coords.ndim == 3:
        coords = coords[0]
    steps = torch.norm(coords[1:] - coords[:-1], dim=-1)
    return (steps.std() / steps.mean()).item()


# Helper: Device check
def get_test_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Helper: Local predictor fallback
class LightweightPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}
        self.embedding = nn.Embedding(21, 16)
        self.linear1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.coord_out = nn.Linear(16, 3)
        self.plddt_out = nn.Linear(16, 1)

    def predict(self, binder_seq: str, target_seq: str) -> np.ndarray:
        L = len(binder_seq)
        indices = [self.char_to_idx.get(c, 20) for c in binder_seq]
        x_idx = torch.tensor(indices, dtype=torch.long)

        # Forward pass for coordinates
        embeds = self.embedding(x_idx)
        h = self.relu(self.linear1(embeds))
        coords_delta = self.coord_out(h)

        # Base helical spiral coords
        t = torch.linspace(0, 4 * math.pi, L).unsqueeze(1)
        base_coords = torch.cat([torch.sin(t) * 2.0, torch.cos(t) * 2.0, t], dim=-1)

        # Combine with coordinate predictions
        coords = base_coords + coords_delta * 0.1

        return coords.unsqueeze(0).detach().numpy()

    def predict_plddt(self, binder_seq: str) -> float:
        indices = [self.char_to_idx.get(c, 20) for c in binder_seq]
        x_idx = torch.tensor(indices, dtype=torch.long)

        embeds = self.embedding(x_idx)
        h = self.relu(self.linear1(embeds))
        plddt_vals = self.plddt_out(h)

        # Dynamically scale average prediction using sigmoid to map to [70, 100]
        mean_plddt = torch.sigmoid(plddt_vals.mean()) * 30.0 + 70.0
        return mean_plddt.item()

def get_predictor():
    if HAS_COREAI and RUN_COREAI_INTEGRATION:
        try:
            return DynamicStructurePredictor()
        except Exception:
            return LightweightPredictor()
    return LightweightPredictor()


# =====================================================================
# TIER 1: FEATURE SPECIFIC FUNCTIONAL TESTS (20 TESTS)
# =====================================================================

# --- Feature 1: MPS Execution (5 tests) ---

def test_t1_f1_float32_casting_compatibility():
    """Verify that inputs cast to float32 perform standard linear calculations without double types."""
    x = torch.randn(4, 4, dtype=torch.float32)
    linear = nn.Linear(4, 4).float()
    out = linear(x)
    assert out.dtype == torch.float32
    assert not any(p.dtype == torch.float64 for p in linear.parameters())

def test_t1_f1_dynamic_autocast_wrappers():
    """Verify that autocast_device_type handles mps/cuda/cpu appropriately."""
    assert autocast_device_type("cpu") == "cpu"
    # Ensure it runs without exception in torch.autocast context wrapper
    device_type = "cpu"
    device_autocast = autocast_device_type(device_type)
    with torch.autocast(device_autocast, enabled=False):
        x = torch.randn(2, 2)
        y = x * 2
    assert y.shape == (2, 2)

def test_t1_f1_device_selection():
    """Verify device selection picks cuda/mps/cpu dynamically and initializes correctly."""
    device = get_test_device()
    assert isinstance(device, torch.device)
    t = torch.randn(2, 2, device=device)
    assert t.device.type == device.type

def test_t1_f1_mps_execution_simulated():
    """Check that layer/group normalization and einsum execute error-free on the available device."""
    device = get_test_device()
    x = torch.randn(2, 8, 8, device=device, dtype=torch.float32)
    # Check LayerNorm
    norm = nn.LayerNorm(8, device=device)
    y = norm(x)
    assert y.shape == x.shape
    # Check Einsum
    a = torch.randn(2, 4, 8, device=device)
    b = torch.randn(2, 8, 4, device=device)
    c = torch.einsum('bij,bjk->bik', a, b)
    assert c.shape == (2, 4, 4)

def test_t1_f1_diffusion_attention_on_mps():
    """Check that attention modules execute correctly under float32 on the selected device."""
    device = get_test_device()
    c_s, c_z, num_heads = 16, 8, 2
    B, N = 1, 8
    
    s = torch.randn(B, N, c_s, device=device, dtype=torch.float32)
    z = torch.randn(B, N, N, c_z, device=device, dtype=torch.float32)
    mask = torch.ones(B, N, device=device, dtype=torch.float32)
    
    attention = AttentionPairBias(c_s, c_z, num_heads, use_mla=False, use_fold_cp=False).to(device).float()
    out = attention(s, z, mask)
    assert out.shape == (B, N, c_s)
    assert out.dtype == torch.float32


# --- Feature 2: Low-Rank Pair Updates (5 tests) ---

def test_t1_f2_low_rank_tensor_product_shape():
    """Verify low-rank tensor product output shape is correct."""
    B, N, d, D_pair = 2, 8, 4, 16
    X = torch.randn(B, N, d)
    Y = torch.randn(B, N, d)
    W = torch.randn(D_pair, d)
    U = LowRankTensorProduct.apply(X, Y, W)
    assert U.shape == (B, N, N, D_pair)

def test_t1_f2_gradient_consistency():
    """Verify gradient computation consistency for the custom autograd function."""
    B, N, d, D_pair = 2, 4, 2, 8
    X = torch.randn(B, N, d, requires_grad=True, dtype=torch.float64)
    Y = torch.randn(B, N, d, requires_grad=True, dtype=torch.float64)
    W = torch.randn(D_pair, d, requires_grad=True, dtype=torch.float64)
    
    # Run torch.autograd.gradcheck
    test_passed = torch.autograd.gradcheck(LowRankTensorProduct.apply, (X, Y, W), eps=1e-6, atol=1e-4)
    assert test_passed

def test_t1_f2_memory_scaling():
    """Verify activation memory footprint of low-rank pair updates vs full-rank.

    This previously compared *parameter* counts, which is not what the low-rank
    factorization targets and not what the checklist claims. Activation retention
    is the real quantity, and it is measurable portably via saved_tensors_hooks.
    """
    d_seq, d_pair, rank = 64, 64, 8
    n_tokens = 96
    seq = torch.randn(1, n_tokens, d_seq, requires_grad=True)

    low_rank_module = LowRankPairUpdater(d_seq, d_pair, rank=rank)
    full_rank_module = FullRankPairUpdater(d_seq, d_pair, d_mid=rank)

    low_saved, low_out = saved_element_count(lambda: low_rank_module(seq))
    full_saved, full_out = saved_element_count(lambda: full_rank_module(seq))

    # Like-for-like: identical output shape, different activation cost.
    assert low_out.shape == full_out.shape == (1, n_tokens, n_tokens, d_pair)
    assert full_saved > 0
    assert low_saved < full_saved, (
        f"low-rank retained {low_saved} elements vs full-rank {full_saved}; "
        f"the factorization is supposed to retain fewer"
    )

    # Parameter count is a separate, weaker property -- keep it, but it is no
    # longer the thing standing in for the activation claim.
    low_rank_params = sum(p.numel() for p in low_rank_module.parameters())
    full_rank_params = sum(p.numel() for p in full_rank_module.parameters())
    assert low_rank_params < full_rank_params

def test_t1_f2_weight_initialization():
    """Verify that weight matrices in LowRankPairUpdater are correctly initialized (not nan or zero)."""
    module = LowRankPairUpdater(d_seq=16, d_pair=32, rank=4)
    assert torch.all(torch.isfinite(module.W))
    assert torch.all(torch.isfinite(module.proj_x.weight))
    # Standard deviation check for initialization
    assert module.W.std() > 0.01

def test_t1_f2_small_tensors_stability():
    """Test LowRankPairUpdater stability with minimum input dimensions."""
    module = LowRankPairUpdater(d_seq=2, d_pair=2, rank=1)
    s = torch.randn(1, 2, 2)
    out = module(s)
    assert out.shape == (1, 2, 2, 2)
    assert not torch.isnan(out).any()


# --- Feature 3: CFG Distillation (5 tests) ---

def test_t1_f3_teacher_student_matching():
    """Verify that student weights are initialized properly from teacher."""
    teacher = TeacherVectorField(node_dim=16, seq_dim=8)
    student = CFGDistilledVectorField(node_dim=16, seq_dim=8)
    initialize_student_from_teacher(student, teacher)
    # Check weights match
    assert torch.allclose(student.coord_proj.weight, teacher.coord_proj.weight)
    assert torch.allclose(student.out_head[0].weight, teacher.out_head[0].weight)

def test_t1_f3_speculative_sampler_lookahead():
    """Check K-step speculative sampler lookahead step proposal."""
    def dummy_vf(x, t, **kwargs):
        return torch.zeros_like(x)
    
    sampler = SpeculativeFlowMatchingSampler(
        draft_vf_fn=dummy_vf,
        target_vf_fn=dummy_vf,
        step_size=0.1,
        speculative_lookahead=3,
        tolerance=0.1
    )
    x_init = torch.randn(1, 4, 3)
    res, stats = sampler.sample(x_init)
    assert stats["total_drafts_proposed"] > 0
    assert res.shape == x_init.shape

def test_t1_f3_guidance_embeddings_dimension():
    """Check dimension of sinusoidal embeddings for time steps and guidance scale."""
    embedder = SinusoidalEmbedding(dim=32)
    s = torch.tensor([1.5, 3.0])
    emb = embedder(s)
    assert emb.shape == (2, 32)

def test_t1_f3_step_acceptance_logic():
    """Verify draft step acceptance rate when draft aligns vs diverges from target."""
    def perfect_draft(x, t, **kwargs):
        return -x
    def target(x, t, **kwargs):
        return -x
    def bad_draft(x, t, **kwargs):
        return x
        
    sampler_perfect = SpeculativeFlowMatchingSampler(
        draft_vf_fn=perfect_draft,
        target_vf_fn=target,
        step_size=0.2,
        speculative_lookahead=2,
        tolerance=0.01
    )
    sampler_bad = SpeculativeFlowMatchingSampler(
        draft_vf_fn=bad_draft,
        target_vf_fn=target,
        step_size=0.2,
        speculative_lookahead=2,
        tolerance=0.01
    )
    x_init = torch.randn(1, 5, 3)
    _, stats_perfect = sampler_perfect.sample(x_init)
    _, stats_bad = sampler_bad.sample(x_init)
    
    assert stats_perfect["acceptance_rate"] == 1.0
    assert stats_bad["acceptance_rate"] < 0.5

def test_t1_f3_student_forward_efficiency():
    """Verify the student produces a guided field in one pass where CFG needs two.

    Previously this asserted only `pred.shape`; the teacher was never invoked and
    the "Profile forward passes count" comment profiled nothing, so the
    single-vs-double claim went unverified. Forward invocations are countable
    without training, so count them.
    """
    student = CFGDistilledVectorField(node_dim=16, seq_dim=8)
    teacher = TeacherVectorField(node_dim=16, seq_dim=8)
    x = torch.randn(1, 4, 3)
    t = torch.tensor([0.5])
    c = torch.randn(1, 4, 8)
    s = torch.tensor([2.0])
    guidance_scale = 2.0

    calls = {"teacher": 0, "student": 0}
    teacher.register_forward_hook(lambda *_: calls.__setitem__("teacher", calls["teacher"] + 1))
    student.register_forward_hook(lambda *_: calls.__setitem__("student", calls["student"] + 1))

    # Classifier-free guidance with the teacher: one conditional pass plus one
    # unconditional pass, combined.
    v_cond = teacher(x, t, c, cond_mask=torch.ones(1))
    v_uncond = teacher(x, t, c, cond_mask=torch.zeros(1))
    guided = v_uncond + guidance_scale * (v_cond - v_uncond)

    # The distilled student takes the guidance scale as an input instead.
    pred = student(x, t, c, s)

    assert calls["teacher"] == 2, f"CFG should need 2 teacher passes, saw {calls['teacher']}"
    assert calls["student"] == 1, f"student should need 1 pass, saw {calls['student']}"
    assert pred.shape == guided.shape == (1, 4, 3)


# --- Feature 4: Neural Refinement (5 tests) ---

def test_t1_f4_resnet_coordinate_refiner_shape():
    """Check refiner coordinate shape preservation."""
    refiner = ResNetCoordinateRefiner(embed_dim=16, hidden_dim=16)
    seq = torch.randn(1, 10, 16)
    coords = torch.randn(1, 10, 3)
    refined = refiner(seq, coords)
    assert refined.shape == coords.shape

def test_t1_f4_supervised_loss_calculation():
    """Verify supervised MSE loss for coordinate and distance matrices."""
    pred = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    true = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    coord_loss, dist_loss = compute_supervised_loss(pred, true)
    
    # Coord loss: mean((pred-true)^2) = mean(0 + (1-2)^2) = 1/6 = 0.1667
    assert math.isclose(coord_loss.item(), 1.0/6.0, abs_tol=1e-4)
    # Distance loss: pred_dists=[[0,1],[1,0]], true_dists=[[0,2],[2,0]]
    # mean((pred_dists-true_dists)^2) = mean(0 + 1 + 1 + 0) = 0.5
    assert math.isclose(dist_loss.item(), 0.5, abs_tol=1e-4)

def test_t1_f4_clash_index_improvement():
    """Verify that a trained refiner minimizes steric clash indexes (minimum non-consecutive distances)."""
    torch.manual_seed(42)
    refiner = ResNetCoordinateRefiner(embed_dim=16, hidden_dim=16)
    seq = torch.randn(1, 10, 16)
    # Deliberately force clash by placing residues 2 and 5 at similar coordinates
    coords = torch.randn(1, 10, 3)
    coords[0, 2] = torch.tensor([1.0, 1.0, 1.0])
    coords[0, 5] = torch.tensor([1.01, 1.01, 1.01])
    
    # Before refinement, min non-consecutive distance is small
    c1 = coords[0].unsqueeze(1)
    c2 = coords[0].unsqueeze(0)
    dists = torch.norm(c1 - c2, dim=-1)
    mask = torch.abs(torch.arange(10).unsqueeze(1) - torch.arange(10).unsqueeze(0)) > 1
    min_dist_before = dists[mask].min().item()
    
    # Run through refiner
    refined = refiner(seq, coords)
    rf1 = refined[0].unsqueeze(1)
    rf2 = refined[0].unsqueeze(0)
    refined_dists = torch.norm(rf1 - rf2, dim=-1)
    min_dist_after = refined_dists[mask].min().item()

    # NOTE: this does not assert clash *improvement*, despite the test name. The
    # refiner here is randomly initialised, so requiring min_dist_after >
    # min_dist_before would be asserting untrained-network luck. What is
    # checkable is that refinement preserves shape and does not collapse
    # non-consecutive residues onto each other. Tightening this to a real
    # improvement check needs a trained refiner checkpoint.
    assert refined.shape == coords.shape
    assert min_dist_before > 0.0
    assert min_dist_after > 0.0, (
        f"refiner collapsed non-consecutive residues: "
        f"min distance {min_dist_before:.4f} -> {min_dist_after:.4f} A"
    )

def test_t1_f4_bond_length_error_correction():
    """Exercise the refiner on a chain with non-physical bond lengths.

    Scope note: this does NOT verify correction toward the ideal 3.8 A CA-CA
    spacing, despite the name and the TEST_READY.md entry. The refiner here is
    randomly initialised, so asserting that it moves bond lengths toward 3.8
    would be asserting luck rather than behaviour. What is checkable without a
    trained checkpoint is that the refiner consumes deliberately non-physical
    geometry and returns a finite, still-connected chain.

    To make this a real correction test, load a trained refiner (see
    train_coordinate_refiner.py / test_refiner_checkpoint_load_roundtrip in
    test_boltz_modified_layers.py) and assert the deviation from 3.8 A shrinks.
    """
    refiner = ResNetCoordinateRefiner(embed_dim=16, hidden_dim=16)
    seq = torch.randn(1, 5, 16)
    # Bond length 5.0 vs the physical ideal of 3.8 A.
    coords = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0], [20.0, 0.0, 0.0]]])
    refined = refiner(seq, coords)

    diffs = refined[0, 1:] - refined[0, :-1]
    dists = torch.norm(diffs, dim=-1)
    assert dists.shape == (4,)
    assert torch.all(torch.isfinite(dists))
    assert torch.all(dists > 0)
    before = (torch.tensor([5.0] * 4) - 3.8).abs().mean().item()
    after = (dists - 3.8).abs().mean().item()
    print(f"[diagnostic] mean |bond - 3.8 A|: {before:.3f} -> {after:.3f} "
          f"(untrained refiner; not asserted)")

def test_t1_f4_refiner_loading_inference(tmp_path):
    """Verify saving/loading of ResNetCoordinateRefiner weights.

    The in-memory state_dict handoff below does not exercise saving, despite the
    name, so a real torch.save/torch.load round-trip through disk is included.
    """
    model1 = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8)
    model2 = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8)
    seq = torch.randn(1, 4, 8)
    coords = torch.randn(1, 4, 3)

    # Differently-initialised models must disagree first, or the checks below
    # would pass even if load_state_dict did nothing.
    assert not torch.allclose(model1(seq, coords), model2(seq, coords))

    model2.load_state_dict(model1.state_dict())
    out1 = model1(seq, coords)
    out2 = model2(seq, coords)
    assert torch.allclose(out1, out2)

    # Round-trip through disk.
    ckpt = tmp_path / "refiner.pt"
    torch.save(model1.state_dict(), ckpt)
    model3 = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8)
    model3.load_state_dict(torch.load(ckpt, weights_only=True))
    assert torch.allclose(model1(seq, coords), model3(seq, coords))


# =====================================================================
# TIER 2: BOUNDARY & CORNER CASES (20 TESTS, 5 PER FEATURE)
# =====================================================================

# --- Feature 1: MPS Execution (5 tests) ---

def test_t2_f1_empty_residue_sequence():
    """Verify behavior with zero sequence size."""
    module = LowRankPairUpdater(d_seq=8, d_pair=16, rank=4)
    s = torch.zeros(1, 0, 8)
    # Should either raise an error or return an empty tensor safely
    try:
        out = module(s)
        # If it doesn't raise, verify the output is empty with correct shape
        assert out.shape == (1, 0, 0, 16)
    except Exception:
        pass  # An exception is also acceptable behavior for empty input

def test_t2_f1_large_sequence_prediction():
    """Verify performance compatibility of float32 tensors with length 1000+ residues."""
    device = get_test_device()
    N = 1000
    x = torch.randn(1, N, 8, device=device, dtype=torch.float32)
    # Test a simple projection mapping
    proj = nn.Linear(8, 8, device=device)
    y = proj(x)
    assert y.shape == (1, 1000, 8)

def test_t2_f1_autocast_enabled_vs_disabled():
    """Verify that enabling vs disabling autocast wrapper runs without crashes."""
    device = get_test_device()
    module = nn.Linear(8, 8, device=device)
    x = torch.randn(2, 8, device=device)
    
    dev_type = "cpu" if device.type == "cpu" else device.type
    dev_autocast = autocast_device_type(dev_type)
    
    with torch.autocast(dev_autocast, enabled=True):
        out_enabled = module(x)
    with torch.autocast(dev_autocast, enabled=False):
        out_disabled = module(x)
        
    assert out_enabled.shape == out_disabled.shape == (2, 8)

def test_t2_f1_device_fallback():
    """Verify fallback device defaults to CPU when invalid device is given."""
    # This was a bare try/except with its assertions inside the except branch, so
    # if torch ever stopped raising, the test would pass having checked nothing.
    # Make the expectation explicit, and assert on the suite's real device
    # selector rather than the tautology torch.device("cpu").type == "cpu".
    with pytest.raises(RuntimeError):
        torch.device("invalid_device_name")
    assert get_test_device().type in {"cpu", "cuda", "mps"}

def test_t2_f1_zero_rank_scalar_tensors():
    """Verify handling of 0D scalar tensors for inputs like time/scale."""
    embedder = SinusoidalEmbedding(dim=16)
    t = torch.tensor(0.5) # scalar
    emb = embedder(t)
    assert emb.shape == (1, 16)


# --- Feature 2: Low-Rank Pair Updates (5 tests) ---

def test_t2_f2_minimum_rank_d1():
    """Verify low rank updates with minimum rank d=1."""
    module = LowRankPairUpdater(d_seq=8, d_pair=16, rank=1)
    s = torch.randn(1, 4, 8)
    out = module(s)
    assert out.shape == (1, 4, 4, 16)

def test_t2_f2_over_complete_rank():
    """Verify low rank updates with over-complete rank d >= C_z."""
    # d_pair = 8, rank = 16 (over-complete rank)
    module = LowRankPairUpdater(d_seq=4, d_pair=8, rank=16)
    s = torch.randn(1, 4, 4)
    out = module(s)
    assert out.shape == (1, 4, 4, 8)

def test_t2_f2_sparse_zero_input_tensors():
    """Verify output and gradients remain zero with sparse zero inputs."""
    B, N, d, D_pair = 1, 4, 2, 8
    X = torch.zeros(B, N, d, requires_grad=True)
    Y = torch.zeros(B, N, d, requires_grad=True)
    W = torch.randn(D_pair, d, requires_grad=True)
    
    U = LowRankTensorProduct.apply(X, Y, W)
    assert torch.all(U == 0.0)
    
    loss = U.sum()
    loss.backward()
    assert torch.all(X.grad == 0.0)
    assert torch.all(Y.grad == 0.0)

def test_t2_f2_inf_nan_gradients():
    """Verify backward safety against infinity gradients."""
    B, N, d, D_pair = 1, 4, 2, 8
    X = torch.randn(B, N, d, requires_grad=True)
    Y = torch.randn(B, N, d, requires_grad=True)
    W = torch.randn(D_pair, d, requires_grad=True)
    
    U = LowRankTensorProduct.apply(X, Y, W)
    # Inject large values
    grad_output = torch.full_like(U, float('inf'))
    U.backward(grad_output)
    # Gradients will contain nans or infs, verify it does not trigger segmentation fault
    assert X.grad is not None
    assert torch.any(torch.isinf(X.grad) | torch.isnan(X.grad))

def test_t2_f2_memory_scaling_limits():
    """Ensure high sequence size (N=2000) does not allocate excessive memory for low rank OPM."""
    d_seq = 32
    rank = 8
    s = torch.randn(1, 2000, d_seq)
    # Check that initializing LowRankPairUpdater remains fast and uses minimal memory
    module = LowRankPairUpdater(d_seq, d_pair=16, rank=rank)
    out = module(s)
    assert out.shape == (1, 2000, 2000, 16)


# --- Feature 3: CFG Distillation (5 tests) ---

def test_t2_f3_guidance_scale_s0():
    """Verify CFG distilled vector field with zero guidance scale."""
    student = CFGDistilledVectorField(node_dim=8, seq_dim=4)
    x = torch.randn(1, 4, 3)
    t = torch.tensor([0.2])
    c = torch.randn(1, 4, 4)
    s = torch.tensor([0.0]) # guidance = 0
    out = student(x, t, c, s)
    assert out.shape == (1, 4, 3)
    assert not torch.isnan(out).any()

def test_t2_f3_negative_extreme_guidance_scale():
    """Verify behavior of scale s under negative or extreme values (e.g. -2.0 or 10.0)."""
    student = CFGDistilledVectorField(node_dim=8, seq_dim=4)
    x = torch.randn(1, 4, 3)
    t = torch.tensor([0.5])
    c = torch.randn(1, 4, 4)
    
    out_neg = student(x, t, c, torch.tensor([-2.0]))
    out_ext = student(x, t, c, torch.tensor([10.0]))
    
    assert out_neg.shape == out_ext.shape == (1, 4, 3)
    assert not torch.isnan(out_neg).any()
    assert not torch.isnan(out_ext).any()

def test_t2_f3_lookahead_size():
    """Test speculative sampler under different lookahead sizes K=1 and K=10."""
    def dummy_vf(x, t, **kwargs):
        return -x
        
    sampler_k1 = SpeculativeFlowMatchingSampler(dummy_vf, dummy_vf, step_size=0.1, speculative_lookahead=1)
    sampler_k10 = SpeculativeFlowMatchingSampler(dummy_vf, dummy_vf, step_size=0.05, speculative_lookahead=10)
    
    x_init = torch.randn(1, 4, 3)
    res_k1, stats_k1 = sampler_k1.sample(x_init)
    res_k10, stats_k10 = sampler_k10.sample(x_init)
    
    # With a smaller step size, K=10 generates more total draft proposals
    assert stats_k1["total_drafts_proposed"] <= stats_k10["total_drafts_proposed"]

def test_t2_f3_step_rejection_100():
    """Verify recovery when all speculative steps are rejected (tolerance = 0)."""
    def draft(x, t, **kwargs):
        return x
    def target(x, t, **kwargs):
        return -x
        
    sampler = SpeculativeFlowMatchingSampler(draft, target, step_size=0.2, speculative_lookahead=3, tolerance=-1.0)
    x_init = torch.randn(1, 4, 3)
    res, stats = sampler.sample(x_init)
    # All drafts rejected
    assert stats["total_drafts_accepted"] == 0
    assert res.shape == x_init.shape

def test_t2_f3_time_boundaries():
    """Check distilled vector field predictions at boundaries t=0.0 and t=1.0."""
    student = CFGDistilledVectorField(node_dim=8, seq_dim=4)
    x = torch.randn(1, 4, 3)
    c = torch.randn(1, 4, 4)
    s = torch.tensor([1.0])
    
    out_t0 = student(x, torch.tensor([0.0]), c, s)
    out_t1 = student(x, torch.tensor([1.0]), c, s)
    
    assert out_t0.shape == out_t1.shape == (1, 4, 3)


# --- Feature 4: Neural Refinement (5 tests) ---

def test_t2_f4_coordinates_nan_zero():
    """Verify coordinate refiner behavior with coordinate tensors containing NaNs or all zeros."""
    refiner = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8)
    seq = torch.randn(1, 4, 8)
    
    # Zero coords
    coords_zero = torch.zeros(1, 4, 3)
    refined_zero = refiner(seq, coords_zero)
    assert refined_zero.shape == (1, 4, 3)
    
    # NaN coords
    coords_nan = torch.full((1, 4, 3), float('nan'))
    refined_nan = refiner(seq, coords_nan)
    assert refined_nan.shape == (1, 4, 3)

def test_t2_f4_multi_chain_boundaries():
    """Test refinement handling of multi-chain boundaries (simulating large distance gaps)."""
    refiner = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8)
    seq = torch.randn(1, 10, 8)
    coords = torch.randn(1, 10, 3)
    # Simulate multi-chain break by adding 100 Angstrom gap at residue index 5
    coords[0, 5:] = coords[0, 5:] + 100.0
    
    refined = refiner(seq, coords)
    assert refined.shape == (1, 10, 3)
    # Verify the gap remains large (not flattened to single center)
    gap = torch.norm(refined[0, 5] - refined[0, 4])
    assert gap > 80.0

def test_t2_f4_perfect_coordinates_refinement():
    """Verify refiner does not distort already perfect/valid coordinates trace."""
    refiner = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8)
    # Make projection layers zero weight output deltas to check identity mapping
    nn.init.zeros_(refiner.proj_delta.weight)
    nn.init.zeros_(refiner.proj_delta.bias)
    
    seq = torch.randn(1, 6, 8)
    coords = generate_mock_ground_truth(6)
    refined = refiner(seq, coords)
    assert torch.allclose(refined, coords, atol=1e-5)

def test_t2_f4_extreme_clashing_coordinates():
    """Verify that refiner moves coordinates that are extremely clashing (same physical point)."""
    refiner = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8)
    seq = torch.randn(1, 4, 8)
    # All residues placed at the exact same location
    coords = torch.zeros(1, 4, 3)
    
    refined = refiner(seq, coords)
    assert refined.shape == (1, 4, 3)
    assert torch.all(torch.isfinite(refined))

    # `assert torch.any(refined != 0.0)` used to stand alone here, which only
    # showed that *something* moved -- not that the coincident residues were
    # separated from each other, which is what the test is named for. They do
    # separate even untrained, because each residue carries a different sequence
    # embedding: measured min pairwise separation over 60 initialisations was
    # 0.0023, never below 1e-4.
    pairwise = torch.cdist(refined[0], refined[0])
    off_diagonal = pairwise[~torch.eye(4, dtype=torch.bool)]
    assert off_diagonal.min() > 1e-5, (
        f"residues starting at the same point were not separated "
        f"(min pairwise distance {off_diagonal.min():.2e})"
    )

def test_t2_f4_residue_length_mismatch():
    """Verify that mismatches in input lengths between coordinates and sequence embeddings raises an error."""
    refiner = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8)
    seq = torch.randn(1, 5, 8)
    coords = torch.randn(1, 4, 3)
    
    with pytest.raises(Exception):
        refiner(seq, coords)


# =====================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (4 TESTS)
# =====================================================================

def test_t3_f1_f2_low_rank_updates_mps():
    """Verify LowRankPairUpdater forward/backward execution on MPS (or fallback CPU) using float32."""
    device = get_test_device()
    module = LowRankPairUpdater(d_seq=8, d_pair=8, rank=2).to(device).float()
    
    s = torch.randn(1, 4, 8, device=device, dtype=torch.float32, requires_grad=True)
    out = module(s)
    loss = out.sum()
    loss.backward()
    
    assert out.shape == (1, 4, 4, 8)
    assert out.dtype == torch.float32
    assert s.grad is not None

def test_t3_f1_f3_speculative_sampler_mps():
    """Verify SpeculativeFlowMatchingSampler execution on MPS (or fallback CPU) using float32."""
    device = get_test_device()
    student = CFGDistilledVectorField(node_dim=8, seq_dim=4).to(device).float()
    teacher = TeacherVectorField(node_dim=8, seq_dim=4).to(device).float()
    
    # Custom wrappers to feed arguments
    c = torch.randn(1, 4, 4, device=device, dtype=torch.float32)
    
    def draft_vf(x, t, **kwargs):
        # Student forward: needs s
        s_tensor = torch.full((x.shape[0],), 1.5, device=device, dtype=torch.float32)
        return student(x, t, c, s_tensor)
        
    def target_vf(x, t, **kwargs):
        # Teacher forward: needs cond_mask
        cond = torch.ones(x.shape[0], device=device, dtype=torch.float32)
        return teacher(x, t, c, cond_mask=cond)
        
    sampler = SpeculativeFlowMatchingSampler(
        draft_vf_fn=draft_vf,
        target_vf_fn=target_vf,
        step_size=0.25,
        speculative_lookahead=2,
        tolerance=0.1
    )
    
    x_init = torch.randn(1, 4, 3, device=device, dtype=torch.float32)
    res, stats = sampler.sample(x_init)
    
    assert res.shape == (1, 4, 3)
    assert res.device.type == device.type
    assert res.dtype == torch.float32

def test_t3_f3_f4_neural_refiner_post_speculative():
    """Verify neural refiner post-processing of coordinates generated by speculative sampler."""
    # 1. Sampler
    def simple_vf(x, t, **kwargs):
        return -x
    sampler = SpeculativeFlowMatchingSampler(simple_vf, simple_vf, step_size=0.2, speculative_lookahead=2)
    x_init = torch.randn(1, 6, 3)
    coarse_coords, _ = sampler.sample(x_init)
    
    # 2. Refinement
    refiner = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8)
    seq = torch.randn(1, 6, 8)
    refined_coords = refiner(seq, coarse_coords)
    
    assert refined_coords.shape == (1, 6, 3)
    assert not torch.isnan(refined_coords).any()

def test_t3_pipeline_integration():
    """Verify E2E execution pipeline: Low-rank updates, speculative flow-matching and coordinate refinement."""
    device = get_test_device()
    
    # Init modules
    pair_updater = LowRankPairUpdater(d_seq=8, d_pair=8, rank=2).to(device).float()
    student = CFGDistilledVectorField(node_dim=8, seq_dim=8).to(device).float()
    teacher = TeacherVectorField(node_dim=8, seq_dim=8).to(device).float()
    refiner = ResNetCoordinateRefiner(embed_dim=8, hidden_dim=8).to(device).float()
    
    seq_embeds = torch.randn(1, 6, 8, device=device, dtype=torch.float32)
    
    # 1. Update pair representations
    z_out = pair_updater(seq_embeds)
    assert z_out.shape == (1, 6, 6, 8)
    
    # 2. Speculative Sample
    def draft_vf(x, t, **kwargs):
        s_tensor = torch.full((x.shape[0],), 2.0, device=device, dtype=torch.float32)
        return student(x, t, seq_embeds, s_tensor)
    def target_vf(x, t, **kwargs):
        cond = torch.ones(x.shape[0], device=device, dtype=torch.float32)
        return teacher(x, t, seq_embeds, cond_mask=cond)
        
    sampler = SpeculativeFlowMatchingSampler(draft_vf, target_vf, step_size=0.25, speculative_lookahead=2)
    x_init = torch.randn(1, 6, 3, device=device, dtype=torch.float32)
    coarse_coords, _ = sampler.sample(x_init)
    
    # 3. Refine
    refined_coords = refiner(seq_embeds, coarse_coords)
    assert refined_coords.shape == (1, 6, 3)
    assert refined_coords.device.type == device.type


# =====================================================================
# TIER 4: REAL-WORLD BIOLOGICAL VALIDATION (5 TESTS)
# =====================================================================

def test_t4_1_human_insulin_monomer():
    """Evaluate structure prediction on Human Insulin Monomer (51 residues) and verify RMSD/pLDDT."""
    insulin_seq = "GIVEQCCTSICSLYQLENYCNFVNQHLCGSHLVEALYLVCGERGFFYTPKT"
    assert len(insulin_seq) == 51
    
    target_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKG"
    
    predictor = get_predictor()
    coords = predictor.predict(insulin_seq, target_seq)
    
    assert coords.shape == (1, 51, 3)
    
    # Compare with a baseline mock structure (ideal spiral fold representation)
    baseline_coords = generate_mock_ground_truth(51)
    rmsd = calculate_rmsd(coords, baseline_coords)
    
    # `assert rmsd < 50.0` used to stand here. It could not distinguish the model
    # from noise: measured 22.16 for the predictor against 25.21-25.63 for
    # torch.randn, so random coordinates satisfied it too. An absolute RMSD bound
    # against generate_mock_ground_truth -- a synthetic 3.8 A helix, not an
    # experimental structure -- cannot be meaningful here, particularly because
    # the fallback predictor emits ~0.56 A CA-CA spacing (see TEST_AUDIT.md E1),
    # so the RMSD is dominated by a scale mismatch rather than by fold accuracy.
    #
    # Assert backbone continuity instead, which does separate signal from noise.
    assert math.isfinite(rmsd)
    continuity = backbone_continuity_cv(coords)
    assert continuity < 0.25, (
        f"backbone spacing is not chain-like (CV={continuity:.4f}); "
        f"random coordinates score above 0.33"
    )
    print(f"[diagnostic] insulin: RMSD vs mock baseline {rmsd:.2f} A, "
          f"backbone CV {continuity:.4f}")
    
    # pLDDT: check the value contract, not a quality threshold.
    #
    # This previously asserted `plddt >= 70.0`, which cannot fail:
    # LightweightPredictor.predict_plddt returns sigmoid(x) * 30.0 + 70.0, whose
    # infimum is exactly 70.0. The old else-branch was worse -- it fabricated a
    # score from a hydrophobic-residue count and asserted on that invented
    # number. With an untrained surrogate no quality threshold is meaningful, so
    # assert what is genuinely checkable and can fail: a finite score inside the
    # valid pLDDT range, and determinism for a fixed input. Both are real
    # constraints on the CoreAI predictor that get_predictor may return.
    if not hasattr(predictor, "predict_plddt"):
        pytest.skip(f"{type(predictor).__name__} exposes no predict_plddt")
    plddt = predictor.predict_plddt(insulin_seq)
    assert math.isfinite(plddt), f"pLDDT is not finite: {plddt}"
    assert 0.0 <= plddt <= 100.0, f"pLDDT outside valid range: {plddt}"
    assert predictor.predict_plddt(insulin_seq) == pytest.approx(plddt), (
        "pLDDT is not deterministic for a fixed sequence"
    )

def test_t4_2_hemoglobin_subunit_alpha():
    """Evaluate Hemoglobin subunit alpha (142 residues) and verify RMSD and latency."""
    hemo_seq = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    assert len(hemo_seq) == 142
    
    target_seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQ"
    
    predictor = get_predictor()
    
    t0 = time.perf_counter()
    coords = predictor.predict(hemo_seq, target_seq)
    latency = time.perf_counter() - t0
    
    assert coords.shape == (1, 142, 3)
    # Real-time inference benchmark constraint: must execute in under 1.0 second
    assert latency < 1.0
    
    baseline_coords = generate_mock_ground_truth(142)
    rmsd = calculate_rmsd(coords, baseline_coords)
    # `assert rmsd < 120.0` used to stand here and was satisfied by random
    # coordinates too (predictor 65.62 vs torch.randn 69.03-71.x), so it verified
    # nothing. Same reasoning as test_t4_1: assert backbone continuity, which
    # separates the two, and keep RMSD as a reported diagnostic.
    assert math.isfinite(rmsd)
    continuity = backbone_continuity_cv(coords)
    assert continuity < 0.25, (
        f"backbone spacing is not chain-like (CV={continuity:.4f}); "
        f"random coordinates score above 0.33"
    )
    print(f"[diagnostic] hemoglobin: RMSD vs mock baseline {rmsd:.2f} A, "
          f"backbone CV {continuity:.4f}")

def test_t4_3_tnf_alpha_complex():
    """Evaluate TNF-alpha complex structure (157 residues) and check clash penalties/PDB parsing."""
    tnf_seq = "VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL"
    assert len(tnf_seq) == 157
    
    # The PDB is an optional external DMS artifact.  Its absence (or a stale
    # placeholder produced by a previous run) must not change this portable
    # functional test's result.
    pdb_path = "/tmp/biomolecular_design/TNF-alpha_1TNF.pdb"
    if os.path.exists(pdb_path):
        # Confirm PDB parsing does not error out
        with open(pdb_path, "r") as f:
            lines = f.readlines()
        assert len(lines) > 0
        atom_lines = [l for l in lines if l.startswith("ATOM")]
        if atom_lines:
            assert all(len(line) >= 54 for line in atom_lines)
        
    predictor = get_predictor()
    coords = predictor.predict(tnf_seq, "MATEVLADIGSAKLR")
    assert coords.shape == (1, 157, 3)
    
    # Check clash penalty (count of non-consecutive residue distances < 2.0 Å)
    coords_tensor = torch.from_numpy(coords)[0]
    c1 = coords_tensor.unsqueeze(1)
    c2 = coords_tensor.unsqueeze(0)
    dists = torch.norm(c1 - c2, dim=-1)
    mask = torch.abs(torch.arange(157).unsqueeze(1) - torch.arange(157).unsqueeze(0)) > 1
    non_consec_dists = dists[mask]
    clashes = torch.sum(non_consec_dists < 2.0).item()

    # Check structure quality: log clashes for diagnostics
    # CoreAI surrogate models are not trained for physical validity,
    # so we only verify the computation runs without error
    total_pairs = non_consec_dists.numel()
    assert total_pairs > 0  # Verify distance matrix was computed
    # The comment above promises diagnostics, so actually emit them rather than
    # computing the count and discarding it.
    print(f"[diagnostic] steric clashes (<2.0 A, non-consecutive): "
          f"{clashes}/{total_pairs} pairs")

def test_t4_4_vegfa_monomer():
    """Evaluate VEGFA monomer structure (110 residues) predicting coordinates and verify exit code (success)."""
    vegf_seq = "APMAEGGGQNHHEVVKFMDVYQRSYCHPIETLVDIFQEYPDEIEYIFKPSCVPLMRCGGCCNDEGLECVPTEESNITMQIMRIKPHQGQHIGEMSFLQHNKCECRPKKDKAR"
    vegf_len = len(vegf_seq)
    assert vegf_len > 100  # VEGFA monomer is > 100 residues
    
    predictor = get_predictor()
    coords = predictor.predict(vegf_seq, "MATEVLAD")
    assert coords.shape == (1, vegf_len, 3)
    assert not torch.isnan(torch.from_numpy(coords)).any()

def test_t4_5_large_scale_validation():
    """Large scale validation (>500 residues) testing latency and activation memory reduction compared to baseline."""
    # Build large sequence of length 512
    large_seq = "MATEVLADIGSAKLR" * 35 # 525 residues
    
    # 1. Verify latency is small (< 1.5s)
    predictor = get_predictor()
    t0 = time.perf_counter()
    coords = predictor.predict(large_seq, "GIVEQCCTS")
    latency = time.perf_counter() - t0
    assert coords.shape == (1, 525, 3)
    assert latency < 1.5
    
    # 2. Activation memory reduction, measured rather than asserted.
    #
    # This previously computed
    #     reduction_pct = (17640000 - 8912) / 17640000 * 100.0
    # from two literals written in the test and asserted it exceeded 30. No
    # tensor or module took part, so the assertion could not fail and counted as
    # coverage for a claim it never checked.
    #
    # saved_tensors_hooks counts what each updater actually retains for backward,
    # which is the quantity the low-rank factorization targets: full-rank
    # materializes an O(N^2 * d_mid) intermediate, low-rank keeps only the
    # O(N * rank) factors.
    d_seq = d_pair = 64
    rank = 8
    reductions = {}
    for n_tokens in (128, 256):
        seq = torch.randn(1, n_tokens, d_seq, requires_grad=True)
        low_rank = LowRankPairUpdater(d_seq, d_pair, rank=rank)
        full_rank = FullRankPairUpdater(d_seq, d_pair, d_mid=rank)

        low_saved, low_out = saved_element_count(lambda: low_rank(seq))
        full_saved, full_out = saved_element_count(lambda: full_rank(seq))

        # Same result shape, so the comparison is like-for-like.
        assert low_out.shape == full_out.shape == (1, n_tokens, n_tokens, d_pair)
        assert full_saved > 0
        reductions[n_tokens] = 100.0 * (1.0 - low_saved / full_saved)

    # Substantial at both sizes...
    for n_tokens, reduction_pct in reductions.items():
        assert reduction_pct > 50.0, (
            f"N={n_tokens}: low-rank retained only {reduction_pct:.2f}% less "
            f"activation memory than full-rank"
        )
    # ...and widening with N, which is the asymptotic claim (O(N^2) vs O(N)).
    # This is what would break if the low-rank path started materializing the
    # quadratic intermediate.
    assert reductions[256] > reductions[128], (
        f"reduction did not grow with sequence length: "
        f"N=128 {reductions[128]:.2f}% vs N=256 {reductions[256]:.2f}%"
    )

def test_t1_f7_adaptive_lookahead_speculative():
    """Verify adaptive lookahead window resizing based on draft acceptance rate."""
    # Dummy vector fields: draft matches target perfectly
    def dummy_vf(x, t, **kwargs):
        return torch.zeros_like(x)
        
    sampler = SpeculativeFlowMatchingSampler(
        draft_vf_fn=dummy_vf,
        target_vf_fn=dummy_vf,
        step_size=0.1,
        speculative_lookahead=2,
        tolerance=0.01,
        adaptive_lookahead=True
    )
    
    x_init = torch.randn(1, 10, 3)
    x_out, stats = sampler.sample(x_init)
    
    assert x_out.shape == x_init.shape
    assert stats["acceptance_rate"] == 1.0
    assert stats["total_drafts_accepted"] == stats["total_drafts_proposed"]

def test_t1_f8_biophysical_manifold_constraint_speculative():
    """Verify that biophysical constraint projection in speculative sampler resolves clashes."""
    # Define a target that does nothing, and draft that does nothing
    def dummy_vf(x, t, **kwargs):
        return torch.zeros_like(x)
        
    sampler = SpeculativeFlowMatchingSampler(
        draft_vf_fn=dummy_vf,
        target_vf_fn=dummy_vf,
        step_size=0.5, # 2 steps
        speculative_lookahead=1,
        enable_biophysical=True
    )
    
    # 2 residues very close (distance = 1.0)
    x_init = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    x_out, stats = sampler.sample(x_init)
    
    # Adjacent C_alpha - C_alpha distance should project towards target_dist (3.80 Å)
    dist_out = torch.norm(x_out[0, 1] - x_out[0, 0]).item()
    assert abs(dist_out - 3.80) < 0.1  # Must project near 3.80 Å

def test_t1_f9_bidirectional_codesign():
    """Verify sequence and structure parameters optimize together in the BidirectionalCoDesigner."""
    model = BidirectionalCoDesigner(seq_len=6)
    target_site = torch.tensor([1.0, 2.0, 3.0])
    loss_fn = JointBiophysicalLoss(target_site=target_site)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Run 3 steps of optimization. Sequence participation is covered by the
    # sequence_logits.grad assertion below, so the pre-optimization sequence is
    # not captured; asserting the decoded string changed would be flaky.
    initial_loss, _ = loss_fn.compute_loss(model(), model.coord_displacements)
    
    for _ in range(3):
        coords = model(temp=0.5)
        loss, _ = loss_fn.compute_loss(coords, model.coord_displacements)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    final_loss, _ = loss_fn.compute_loss(model(), model.coord_displacements)
    
    # Verify parameter updates
    assert final_loss.item() < initial_loss.item()
    # Gradient checks
    assert model.sequence_logits.grad is not None
    assert model.coord_displacements.grad is not None

def test_t1_f10_quantization_aware_training():
    """Verify DynamicQuantizedLinear forward pass, straight-through estimation, and bitwidth optimization."""
    layer = DynamicQuantizedLinear(in_features=16, out_features=16, bias=True, block_size=8, mode='mixed')
    
    # Verify forward works
    x = torch.randn(2, 4, 16)
    out = layer(x)
    assert out.shape == (2, 4, 16)
    
    # Verify average bitwidth is differentiable and between 4 and 8
    avg_bit = layer.get_average_bitwidth()
    assert avg_bit >= 4.0
    assert avg_bit <= 8.0
    
    # Run backward step to verify gradients flow back to full precision weight
    loss = out.sum() + avg_bit
    loss.backward()
    
    assert layer.weight.grad is not None
    assert layer.meta_net[0].weight.grad is not None
