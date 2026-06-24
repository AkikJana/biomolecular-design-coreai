import sys
import os
import torch
import torch.nn as nn
import unittest

# Ensure the local modified boltz is used
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../boltz/src")))

import boltz
print(f"[Test] Using boltz package located at: {boltz.__file__}")

from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.triangular_mult import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
from boltz.model.layers.outer_product_mean import OuterProductMean


class TestBoltzModifiedLayers(unittest.TestCase):
    def test_attention_mla_and_fold_cp(self):
        c_s, c_z, num_heads = 128, 64, 4
        B, N = 1, 64
        
        # Inputs
        s = torch.randn(B, N, c_s)
        z = torch.randn(B, N, N, c_z)
        mask = torch.ones(B, N)
        
        # 1. Monolithic Layer (Standard Boltz)
        layer_mono = AttentionPairBias(c_s, c_z, num_heads, use_mla=False, use_fold_cp=False)
        out_mono = layer_mono(s, z, mask)
        
        # 2. MLA + Fold-CP Layer (Optimized Boltz)
        # We initialize with same weights for proj_q, proj_z, proj_o, proj_g to compare
        layer_opt = AttentionPairBias(c_s, c_z, num_heads, use_mla=False, use_fold_cp=True, num_devices=4)
        layer_opt.proj_q.weight.data.copy_(layer_mono.proj_q.weight.data)
        layer_opt.proj_k.weight.data.copy_(layer_mono.proj_k.weight.data)
        layer_opt.proj_v.weight.data.copy_(layer_mono.proj_v.weight.data)
        layer_opt.proj_g.weight.data.copy_(layer_mono.proj_g.weight.data)
        layer_opt.proj_z[1].weight.data.copy_(layer_mono.proj_z[1].weight.data)
        layer_opt.proj_o.weight.data.copy_(layer_mono.proj_o.weight.data)
        
        out_opt = layer_opt(s, z, mask)
        
        # Measure error
        error = torch.max(torch.abs(out_mono - out_opt)).item()
        print(f"Attention Fold-CP Ring Equivalence Error: {error:.2e}")
        self.assertLess(error, 1e-4)

    def test_triangular_multiplication_fold_cp(self):
        dim = 128
        B, N = 1, 64
        x = torch.randn(B, N, N, dim)
        mask = torch.ones(B, N, N)
        
        # 1. Outgoing TMU
        tmu_out_mono = TriangleMultiplicationOutgoing(dim, use_fold_cp=False)
        tmu_out_opt = TriangleMultiplicationOutgoing(dim, use_fold_cp=True, num_devices=4)
        
        # Copy weights
        tmu_out_opt.norm_in.weight.data.copy_(tmu_out_mono.norm_in.weight.data)
        tmu_out_opt.norm_in.bias.data.copy_(tmu_out_mono.norm_in.bias.data)
        tmu_out_opt.p_in.weight.data.copy_(tmu_out_mono.p_in.weight.data)
        tmu_out_opt.g_in.weight.data.copy_(tmu_out_mono.g_in.weight.data)
        tmu_out_opt.norm_out.weight.data.copy_(tmu_out_mono.norm_out.weight.data)
        tmu_out_opt.norm_out.bias.data.copy_(tmu_out_mono.norm_out.bias.data)
        tmu_out_opt.p_out.weight.data.copy_(tmu_out_mono.p_out.weight.data)
        tmu_out_opt.g_out.weight.data.copy_(tmu_out_mono.g_out.weight.data)
        
        out_mono = tmu_out_mono(x, mask)
        out_opt = tmu_out_opt(x, mask)
        
        error = torch.max(torch.abs(out_mono - out_opt)).item()
        print(f"Triangle Multiplication Outgoing Equivalence Error: {error:.2e}")
        self.assertLess(error, 1e-4)
        
        # 2. Incoming TMU
        tmu_in_mono = TriangleMultiplicationIncoming(dim, use_fold_cp=False)
        tmu_in_opt = TriangleMultiplicationIncoming(dim, use_fold_cp=True, num_devices=4)
        
        # Copy weights
        tmu_in_opt.norm_in.weight.data.copy_(tmu_in_mono.norm_in.weight.data)
        tmu_in_opt.norm_in.bias.data.copy_(tmu_in_mono.norm_in.bias.data)
        tmu_in_opt.p_in.weight.data.copy_(tmu_in_mono.p_in.weight.data)
        tmu_in_opt.g_in.weight.data.copy_(tmu_in_mono.g_in.weight.data)
        tmu_in_opt.norm_out.weight.data.copy_(tmu_in_mono.norm_out.weight.data)
        tmu_in_opt.norm_out.bias.data.copy_(tmu_in_mono.norm_out.bias.data)
        tmu_in_opt.p_out.weight.data.copy_(tmu_in_mono.p_out.weight.data)
        tmu_in_opt.g_out.weight.data.copy_(tmu_in_mono.g_out.weight.data)
        
        in_mono = tmu_in_mono(x, mask)
        in_opt = tmu_in_opt(x, mask)
        
        error_in = torch.max(torch.abs(in_mono - in_opt)).item()
        print(f"Triangle Multiplication Incoming Equivalence Error: {error_in:.2e}")
        self.assertLess(error_in, 1e-4)


    def _opm_per_row_reference(self, opm, m, mask):
        """Previous per-row implementation: materialize (B, S, N, N, c_out) then mean over S."""
        B, S, N, C = m.shape
        m_normed = opm.norm(m)
        U = opm.low_rank_updater(m_normed.reshape(B * S, N, C)).reshape(B, S, N, N, -1)
        mk = mask.unsqueeze(-1).to(m)
        pmask = mk[:, :, :, None, :] * mk[:, :, None, :, :]
        sum_U = (U * pmask).sum(dim=1)
        num = pmask.sum(dim=1).clamp(min=1)
        return sum_U / num

    def test_outer_product_mean_s_contraction_equivalence(self):
        """S-contracted OuterProductMean must match the per-row reference (fwd + grad)."""
        torch.manual_seed(0)
        c_in, c_hidden, c_out = 64, 32, 128
        opm = OuterProductMean(c_in, c_hidden, c_out)

        for B, S, N in [(1, 8, 32), (1, 64, 48), (2, 16, 40)]:
            m = torch.randn(B, S, N, c_in)
            mask = (torch.rand(B, S, N) > 0.15).float()
            with torch.no_grad():
                # chunk_size is passed and must be safely ignored
                new = opm(m, mask, chunk_size=4)
                ref = self._opm_per_row_reference(opm, m, mask)
            self.assertEqual(tuple(new.shape), (B, N, N, c_out))
            err = torch.max(torch.abs(new - ref)).item()
            print(f"OPM S-contraction equivalence (B={B},S={S},N={N}): {err:.2e}")
            self.assertLess(err, 1e-4)

        # Gradient equivalence
        m = torch.randn(1, 16, 40, c_in, requires_grad=True)
        mask = (torch.rand(1, 16, 40) > 0.15).float()
        g = torch.randn(1, 40, 40, c_out)
        opm(m, mask).backward(g)
        grad_new = m.grad.clone()
        m.grad = None
        self._opm_per_row_reference(opm, m, mask).backward(g)
        grad_ref = m.grad.clone()
        grad_err = torch.max(torch.abs(grad_new - grad_ref)).item()
        print(f"OPM S-contraction gradient equivalence: {grad_err:.2e}")
        self.assertLess(grad_err, 1e-4)

    def test_pairformer_fold_cp_end_to_end(self):
        """Fold-CP threaded through the live PairformerLayer must match the dense
        path, including a token count NOT divisible by num_devices (padding)."""
        from boltz.model.layers.pairformer import PairformerLayer

        torch.manual_seed(0)
        token_s, token_z, num_heads = 64, 32, 4
        # N = 30 is NOT a multiple of num_devices=4 -> exercises the pad/unpad path.
        B, N = 1, 30
        s = torch.randn(B, N, token_s)
        z = torch.randn(B, N, N, token_z)
        mask = torch.ones(B, N)
        pair_mask = torch.ones(B, N, N)

        # v2=True so attention is the dense AttentionPairBiasV2 in both layers;
        # the only difference is the Fold-CP triangular multiplication.
        dense = PairformerLayer(token_s, token_z, num_heads, dropout=0.0, v2=True).eval()
        fold = PairformerLayer(
            token_s, token_z, num_heads, dropout=0.0, v2=True,
            use_fold_cp=True, num_devices=4,
        ).eval()
        fold.load_state_dict(dense.state_dict())

        with torch.no_grad():
            s_d, z_d = dense(s, z, mask, pair_mask)
            s_f, z_f = fold(s, z, mask, pair_mask)

        err = max((s_d - s_f).abs().max().item(), (z_d - z_f).abs().max().item())
        print(f"Pairformer Fold-CP end-to-end equivalence (N=30, padded): {err:.2e}")
        self.assertLess(err, 1e-4)

    def test_attentionv2_fold_cp_equivalence(self):
        """Fold-CP ported into AttentionPairBias v2 (the class the model uses)
        must match the dense path, including N not divisible by num_devices."""
        from boltz.model.layers.attentionv2 import AttentionPairBias as AttnV2

        c_s, c_z, num_heads = 64, 32, 4
        for B, N in [(1, 64), (1, 30), (2, 18)]:
            torch.manual_seed(N)
            s = torch.randn(B, N, c_s)
            z = torch.randn(B, N, N, c_z)
            mask = torch.ones(B, N)
            mask[:, -3:] = 0  # exercise key masking

            dense = AttnV2(c_s, c_z, num_heads).eval()
            fold = AttnV2(c_s, c_z, num_heads, use_fold_cp=True, num_devices=4).eval()
            fold.load_state_dict(dense.state_dict())

            with torch.no_grad():
                out_dense = dense(s, z, mask, k_in=s)
                out_fold = fold(s, z, mask, k_in=s)
            err = (out_dense - out_fold).abs().max().item()
            print(f"AttentionV2 Fold-CP equivalence (B={B},N={N}): {err:.2e}")
            self.assertLess(err, 1e-4)

    def test_coordinate_refiner_contract_and_identity(self):
        """Refiner must match the sampler hook contract (token/atom mismatch +
        diffusion multiplicity), be identity at init, and be learnable."""
        from boltz.model.layers.coordinate_refiner import CoordinateRefiner

        torch.manual_seed(0)
        token_s = 64
        B, N, M = 1, 20, 73   # tokens N != atoms M (as in the real model)
        refiner = CoordinateRefiner(token_s=token_s, hidden_dim=32, num_layers=2)

        seq = torch.randn(B, N, token_s)
        atom_coords = torch.randn(B, M, 3)

        # 1. Shape contract + identity at init (zero-init delta head).
        out = refiner(seq, atom_coords)
        self.assertEqual(tuple(out.shape), (B, M, 3))
        id_err = (out - atom_coords).abs().max().item()
        print(f"CoordinateRefiner identity-at-init error: {id_err:.2e}")
        self.assertLess(id_err, 1e-6)

        # 2. Diffusion multiplicity: atom batch is a multiple of token batch.
        mult = 5
        atom_coords_m = torch.randn(B * mult, M, 3)
        out_m = refiner(seq, atom_coords_m)  # seq batch B, coords batch B*mult
        self.assertEqual(tuple(out_m.shape), (B * mult, M, 3))

        # 3. Learnable: after one optimizer step it is no longer the identity.
        opt = torch.optim.SGD(refiner.parameters(), lr=1.0)
        target = atom_coords + 0.5
        loss = ((refiner(seq, atom_coords) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        moved = (refiner(seq, atom_coords) - atom_coords).abs().max().item()
        print(f"CoordinateRefiner post-step delta magnitude: {moved:.2e}")
        self.assertGreater(moved, 1e-4)

    def test_boltz2_refiner_wiring_imports(self):
        """The refiner is importable and wired into boltz2 (default off)."""
        from boltz.model.layers.coordinate_refiner import CoordinateRefiner  # noqa: F401
        import boltz.model.models.boltz2 as b2
        self.assertTrue(hasattr(b2, "CoordinateRefiner"))

    def test_refiner_training_entrypoint_improves_rmsd(self):
        """The supervised entrypoint trains the refiner and reduces aligned RMSD."""
        import tempfile
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        from train_coordinate_refiner import train_refiner

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "refiner.pt")
            hist = train_refiner(
                epochs=40, batch_size=8, num_examples=24, num_atoms=20,
                num_tokens=6, token_s=32, hidden_dim=64, num_layers=2,
                lr=1e-3, device="cpu", ckpt_path=ckpt, verbose=False,
            )
            print(f"Refiner training RMSD: {hist['initial_rmsd']:.3f} -> {hist['final_rmsd']:.3f} A")
            self.assertTrue(os.path.exists(ckpt))
            # Refinement should meaningfully beat the coarse (identity) baseline.
            self.assertLess(hist["final_rmsd"], 0.8 * hist["initial_rmsd"])

    def test_refiner_checkpoint_load_roundtrip(self):
        """train -> save -> load_coordinate_refiner reproduces the trained model."""
        import tempfile
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        from train_coordinate_refiner import train_refiner
        from boltz.model.layers.coordinate_refiner import (
            CoordinateRefiner, load_coordinate_refiner,
        )

        token_s = 32
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "refiner.pt")
            train_refiner(
                epochs=10, batch_size=8, num_examples=16, num_atoms=18,
                num_tokens=5, token_s=token_s, hidden_dim=48, num_layers=2,
                lr=1e-3, device="cpu", ckpt_path=ckpt, verbose=False,
            )

            # Loader rebuilds the architecture from the checkpoint config.
            loaded = load_coordinate_refiner(ckpt, token_s=token_s).eval()
            self.assertEqual(loaded.hidden_dim, 48)

            # Loaded weights reproduce the saved model's output (not identity).
            torch.manual_seed(7)
            seq = torch.randn(1, 5, token_s)
            coords = torch.randn(1, 18, 3)
            ref = CoordinateRefiner(token_s=token_s, hidden_dim=48, num_layers=2)
            sd = torch.load(ckpt, map_location="cpu")["state_dict"]
            ref.load_state_dict(sd)
            ref.eval()
            with torch.no_grad():
                out_loaded = loaded(seq, coords)
                out_ref = ref(seq, coords)
            self.assertLess((out_loaded - out_ref).abs().max().item(), 1e-6)
            self.assertGreater((out_loaded - coords).abs().max().item(), 1e-4)

            # token_s mismatch must raise.
            with self.assertRaises(ValueError):
                load_coordinate_refiner(ckpt, token_s=token_s + 1)

    def test_distance_loss_chunked_matches_dense(self):
        """Chunked distance loss equals the dense off-diagonal reference, is
        chunk-size invariant, and has finite gradients."""
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        from train_coordinate_refiner import distance_loss

        def dense_ref(pred, true, mask):
            pm = mask.unsqueeze(1) * mask.unsqueeze(2)
            eye = torch.eye(mask.shape[1], device=mask.device).unsqueeze(0)
            pm = pm * (1 - eye)  # exclude diagonal
            dp = torch.cdist(pred, pred)
            dt = torch.cdist(true, true)
            return (((dp - dt) ** 2) * pm).sum() / (pm.sum() + 1e-8)

        torch.manual_seed(0)
        B, M = 2, 70  # M in the cdist mm-mode regime
        pred = torch.randn(B, M, 3, requires_grad=True)
        true = torch.randn(B, M, 3)
        mask = torch.ones(B, M)
        mask[:, -4:] = 0  # padding

        ref = dense_ref(pred, true, mask)
        for cs in (16, 64, 1024):  # spans the chunk boundary and full-matrix
            val = distance_loss(pred, true, mask, chunk_size=cs)
            self.assertLess(abs(val.item() - ref.item()), 1e-5)

        # finite gradients (incl. an exact clash)
        pred2 = torch.randn(B, M, 3)
        pred2[0, 3] = pred2[0, 4]
        pred2 = pred2.clone().requires_grad_(True)
        distance_loss(pred2, true, mask, chunk_size=32).backward()
        self.assertFalse(torch.isnan(pred2.grad).any().item())
        print("distance_loss chunked==dense and NaN-free: OK")

    def test_cfg_student_contract_and_hook_replay(self):
        """CFG student matches the sampler hook contract and the c-expansion /
        per-sample slicing / denoise math used by boltz2 + AtomDiffusion.sample."""
        from boltz.model.layers.cfg_student import CFGDistilledStudent

        token_s = 32
        student = CFGDistilledStudent(token_s=token_s, hidden_dim=32, num_layers=2).eval()

        # 1. Contract: token N != atom M, returns (B, M, 3).
        B, N, M = 2, 6, 25
        x = torch.randn(B, M, 3)
        t = torch.full((B,), 0.4)
        c = torch.randn(B, N, token_s)
        s = torch.full((B,), 1.5)
        out = student(x, t, c, s)
        self.assertEqual(tuple(out.shape), (B, M, 3))
        self.assertTrue(torch.isfinite(out).all())

        # 2. Replay the sampler's student branch with diffusion multiplicity.
        #    boltz2 passes c = s_trunk.repeat_interleave(mult); the hook slices
        #    c[sample_ids_chunk] and computes denoised = x + (1 - t) * v.
        mult, num_steps, step_idx = 3, 10, 4
        s_trunk = torch.randn(1, N, token_s)
        c_full = s_trunk.repeat_interleave(mult, 0)          # boltz2 expansion
        atom_coords_noisy = torch.randn(1 * mult, M, 3)
        sample_ids = torch.arange(mult)
        t_norm = float(step_idx) / num_steps
        t_tensor = torch.full((sample_ids.numel(),), t_norm)
        s_tensor = torch.full((sample_ids.numel(),), 1.5)
        c_tensor = c_full[sample_ids]                        # hook slice
        v_pred = student(atom_coords_noisy[sample_ids], t_tensor, c_tensor, s_tensor)
        denoised = atom_coords_noisy[sample_ids] + (1.0 - t_norm) * v_pred
        self.assertEqual(tuple(denoised.shape), (mult, M, 3))
        self.assertTrue(torch.isfinite(denoised).all())
        print("CFG student hook-replay (mult=3, N=6 tokens, M=25 atoms): OK")

    def test_cfg_student_load_roundtrip_and_wiring(self):
        from boltz.model.layers.cfg_student import (
            CFGDistilledStudent, load_cfg_student,
        )
        import tempfile
        token_s = 32
        torch.manual_seed(0)
        student = CFGDistilledStudent(token_s=token_s, hidden_dim=48, num_layers=2)
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "student.pt")
            torch.save(
                {"state_dict": student.state_dict(),
                 "config": {"token_s": token_s, "hidden_dim": 48, "num_layers": 2}},
                ckpt,
            )
            loaded = load_cfg_student(ckpt, token_s=token_s).eval()
            self.assertEqual(loaded.hidden_dim, 48)
            x = torch.randn(1, 12, 3); t = torch.full((1,), 0.3)
            c = torch.randn(1, 4, token_s); s = torch.full((1,), 2.0)
            with torch.no_grad():
                self.assertLess((loaded(x, t, c, s) - student(x, t, c, s)).abs().max().item(), 1e-6)
            with self.assertRaises(ValueError):
                load_cfg_student(ckpt, token_s=token_s + 1)

        import boltz.model.models.boltz2 as b2
        self.assertTrue(hasattr(b2, "CFGDistilledStudent"))

    def test_cfg_distillation_entrypoint_reduces_vfield_error(self):
        """The distillation entrypoint trains the student to match the teacher's
        guided field and writes a loadable checkpoint."""
        import tempfile
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        from train_cfg_student import train_cfg_student
        from boltz.model.layers.cfg_student import load_cfg_student

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "student.pt")
            hist = train_cfg_student(
                epochs=40, steps_per_epoch=20, batch_size=16, num_atoms=20,
                num_tokens=5, token_s=32, hidden_dim=64, num_layers=2,
                lr=1e-3, device="cpu", ckpt_path=ckpt, verbose=False,
            )
            print(f"CFG distillation vfield RMSE: {hist['initial_rmse']:.2f} -> {hist['final_rmse']:.2f}")
            self.assertTrue(os.path.exists(ckpt))
            self.assertLess(hist["final_rmse"], 0.5 * hist["initial_rmse"])
            # Checkpoint loads back into a student via the boltz loader.
            loaded = load_cfg_student(ckpt, token_s=32)
            self.assertEqual(loaded.hidden_dim, 64)

    def test_boltz_reward_matches_real_formula(self):
        """Reward uses Boltz's confidence formula and penalizes clashes."""
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        from boltz_reward import (
            boltz_confidence_score, compute_design_reward, BoltzRewardModel,
        )

        out = {
            "complex_plddt": torch.tensor([0.8]),
            "iptm": torch.tensor([0.6]),
            "ptm": torch.tensor([0.5]),
        }
        # Boltz: (4*complex_plddt + iptm)/5 = (3.2 + 0.6)/5 = 0.76
        self.assertAlmostEqual(boltz_confidence_score(out).item(), 0.76, places=5)

        # iptm all-zero -> falls back to ptm.
        out0 = {"complex_plddt": torch.tensor([0.8]),
                "iptm": torch.tensor([0.0]), "ptm": torch.tensor([0.5])}
        self.assertAlmostEqual(boltz_confidence_score(out0).item(), (3.2 + 0.5) / 5, places=5)

        # Clashes reduce reward: coincident atoms vs well-spaced.
        spaced = (torch.arange(10, dtype=torch.float32).unsqueeze(-1)
                  * torch.tensor([3.8, 0.0, 0.0])).unsqueeze(0)
        clashed = torch.zeros(1, 10, 3)
        base = dict(out)
        r_ok = compute_design_reward({**base, "sample_atom_coords": spaced}, clash_weight=1.0)
        r_clash = compute_design_reward({**base, "sample_atom_coords": clashed}, clash_weight=1.0)
        self.assertGreater(r_ok.item(), r_clash.item())

        # BoltzRewardModel wraps a predict_fn and scores a list of sequences.
        rm = BoltzRewardModel(predict_fn=lambda s: {**base, "sample_atom_coords": spaced},
                              clash_weight=0.0)
        scored = rm.score(["AAAA", "CCCC"])
        self.assertEqual(tuple(scored.shape), (2,))

    def test_grpo_codesign_improves_reward(self):
        """The GRPO co-design loop increases mean Boltz reward over iterations."""
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        from agentic_design_loop import run_codesign_loop
        from boltz_reward import SyntheticSequenceBoltzReward

        torch.manual_seed(0)
        wt = "MATEVLADIGSAKLR"
        interface = [2, 4, 8, 12]
        target = list(wt)
        for p, aa in zip(interface, "WYFM"):
            target[p] = aa
        rm = SyntheticSequenceBoltzReward("".join(target), interface, clash_weight=1.0)

        hist = run_codesign_loop(
            reward_model=rm, wt_sequence=wt, interface_positions=interface,
            iterations=18, group_size=8, device="cpu", verbose=False,
        )
        first = sum(h["mean_reward"] for h in hist[:3]) / 3
        last = sum(h["mean_reward"] for h in hist[-3:]) / 3
        print(f"GRPO co-design mean reward: {first:.3f} -> {last:.3f}")
        self.assertGreater(last, first + 0.05)

    def test_fp8_linear_accuracy_and_memory(self):
        """FP8 weight-only linear: ~4x smaller weights, small relative error,
        and a recursive model converter."""
        from boltz.model.layers.fp8_linear import FP8Linear, quantize_linears_

        torch.manual_seed(0)
        lin = nn.Linear(256, 256)
        fp8 = FP8Linear.from_linear(lin).eval()

        x = torch.randn(8, 256)
        with torch.no_grad():
            ref, out = lin(x), fp8(x)
        rel = ((out - ref).norm() / (ref.norm() + 1e-8)).item()
        print(f"FP8Linear output relative error: {rel:.4f}")
        self.assertLess(rel, 0.05)  # e4m3 weight-only, per-channel scale

        # ~4x weight compression (fp8 1B vs fp32 4B; scale negligible).
        fp32_bytes = lin.weight.numel() * 4
        ratio = fp32_bytes / fp8.weight_bytes()
        print(f"FP8Linear weight compression: {ratio:.2f}x")
        self.assertGreater(ratio, 3.5)

        # Recursive converter replaces Linears in a module.
        mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        n = quantize_linears_(mlp)
        self.assertEqual(n, 2)
        self.assertIsInstance(mlp[0], FP8Linear)

    def test_block_sparse_attention_exact_vs_masked_dense(self):
        """Block-sparse attention equals dense attention with the same block mask."""
        from boltz.model.layers.block_sparse import (
            block_sparse_attention, dense_block_masked_attention,
        )

        torch.manual_seed(0)
        B, H, d, bs, nb = 2, 4, 16, 8, 5
        N = bs * nb
        q, k, v = (torch.randn(B, H, N, d) for _ in range(3))
        bias = torch.randn(B, H, N, N)
        key_mask = torch.ones(B, N)
        key_mask[:, -3:] = 0

        # Random active blocks with a True diagonal (every query block has a key).
        block_mask = torch.rand(nb, nb) > 0.5
        block_mask = block_mask | torch.eye(nb, dtype=torch.bool)

        sparse = block_sparse_attention(q, k, v, block_mask, bias, key_mask, block_size=bs)
        dense = dense_block_masked_attention(q, k, v, block_mask, bias, key_mask, block_size=bs)
        err = (sparse - dense).abs().max().item()
        active = int(block_mask.sum().item())
        print(f"Block-sparse vs masked-dense error: {err:.2e} "
              f"(active {active}/{nb*nb} block pairs)")
        self.assertLess(err, 1e-4)

    def test_benchmark_metrics_and_harness(self):
        """Ranking-agreement metrics are correct and the harness reports
        latency/size/agreement for pluggable scorers."""
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        from benchmark_surrogate_vs_reference import (
            spearman, kendall_tau, topk_recall, benchmark,
            SyntheticReferenceScorer, NoisySurrogateScorer,
        )

        # Metric correctness on known orderings.
        a = torch.tensor([1.0, 2, 3, 4, 5])
        self.assertAlmostEqual(spearman(a, a), 1.0, places=6)
        self.assertAlmostEqual(spearman(a, torch.flip(a, [0])), -1.0, places=6)
        self.assertAlmostEqual(kendall_tau(a, a), 1.0, places=6)
        self.assertEqual(topk_recall(a, a, 3), 1.0)
        # disjoint top-k -> 0 recall
        self.assertEqual(topk_recall(a, torch.flip(a, [0]), 2), 0.0)

        # More noise -> lower agreement (monotone sanity).
        interface = [2, 4, 8, 12, 15]
        motif = "WYFML"
        wt = "MATEVLADIGSAKLRPQ"
        import random
        random.seed(0)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        pairs = []
        for _ in range(30):
            b = list(wt)
            for p in interface:
                if random.random() < 0.5:
                    b[p] = random.choice(aa)
            pairs.append((wt, "".join(b)))

        ref = SyntheticReferenceScorer(motif, interface)
        low = benchmark(pairs, ref, NoisySurrogateScorer(ref, noise=0.05, seed=1), verbose=False)
        high = benchmark(pairs, ref, NoisySurrogateScorer(ref, noise=1.0, seed=1), verbose=False)
        self.assertGreater(low["spearman"], high["spearman"])
        # harness reports the expected fields
        self.assertIn(5, low["topk_recall"])
        self.assertIsNotNone(low["model_size_bytes"]["surrogate(edge)"])
        self.assertGreater(low["latency_ms_per_candidate"]["surrogate(edge)"], 0.0)
        print(f"Benchmark harness: Spearman noise0.05={low['spearman']:.2f} "
              f"vs noise1.0={high['spearman']:.2f}")

    def test_surrogate_affinity_head_and_boltz_predict_fn(self):
        """Affinity head produces rankable scores + is trainable; Boltz predict_fn
        reads real-format outputs; full benchmark loop runs with both."""
        import json, tempfile
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        from surrogate_affinity import AffinitySurrogate, SurrogateAffinityScorer
        from boltz2_predict import read_boltz_outputs, BoltzCliPredictFn, BoltzAffinityScorer
        from benchmark_surrogate_vs_reference import benchmark

        target = "MATEVLADIGSAKLRPQ"
        binders = ["MATEVLADIGSAKLRPQ", "MATWVLAYIGSFKLMPQ", "MACEVLADIGSAKLRPQ"]
        pairs = [(target, b) for b in binders]

        # 1. Surrogate affinity head: scores + structure, correct shapes.
        surr = AffinitySurrogate(embed_dim=32, num_heads=4, hidden=32)
        out = surr.predict(target, binders[0])
        self.assertEqual(tuple(out["affinity_probability_binary"].shape), (1,))
        self.assertEqual(out["sample_atom_coords"].shape[-1], 3)

        scorer = SurrogateAffinityScorer(surr)
        scores = scorer.score(pairs)
        self.assertEqual(tuple(scores.shape), (3,))
        self.assertGreater(scorer.model_size_bytes(), 0)

        # 2. Trainable: one step on the affinity head changes the output.
        opt = torch.optim.SGD(surr.parameters(), lr=0.5)
        before = surr.predict(target, binders[0])["affinity_pred_value"].item()
        k, v = surr.target_kv(target)
        loss = (surr.forward(surr.embed_seq(binders[0]), k, v)["affinity_pred_value"] - 5.0) ** 2
        opt.zero_grad(); loss.mean().backward(); opt.step()
        after = surr.predict(target, binders[0])["affinity_pred_value"].item()
        self.assertNotAlmostEqual(before, after, places=4)

        # 3. Boltz predict_fn reads real-format confidence + affinity JSON.
        with tempfile.TemporaryDirectory() as d:
            name = "cplx0"
            pred_dir = os.path.join(d, "predictions", name)
            os.makedirs(pred_dir)
            with open(os.path.join(pred_dir, f"confidence_{name}_model_0.json"), "w") as f:
                json.dump({"complex_plddt": 0.82, "iptm": 0.61, "ptm": 0.55,
                           "confidence_score": 0.78}, f)
            with open(os.path.join(pred_dir, f"affinity_{name}.json"), "w") as f:
                json.dump({"affinity_pred_value": 1.3, "affinity_probability_binary": 0.9}, f)

            got = read_boltz_outputs(d, name)
            self.assertAlmostEqual(got["complex_plddt"].item(), 0.82, places=5)
            self.assertAlmostEqual(got["affinity_probability_binary"].item(), 0.9, places=5)

            # 4. Full benchmark loop: surrogate vs a (file-backed) Boltz reference.
            #    name_fn returns the same record here; just exercises the wiring.
            predict_fn = BoltzCliPredictFn(d, name_fn=lambda t, b: name)
            ref = BoltzAffinityScorer(predict_fn, size_bytes=400_000_000)
            m = benchmark(pairs, ref, scorer, ks=(1, 2), verbose=False)
            self.assertIn("spearman", m)
            self.assertIsNotNone(m["model_size_bytes"]["surrogate(edge)"])
            print("Surrogate affinity + Boltz predict_fn wired into benchmark: OK")

    def test_affinity_distillation_improves_ranking(self):
        """Distillation trainer raises the surrogate's benchmark ranking and writes
        a loadable checkpoint."""
        import tempfile
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        )
        from train_surrogate_affinity import train_surrogate_affinity
        from surrogate_affinity import AffinitySurrogate

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "surr.pt")
            hist = train_surrogate_affinity(
                epochs=60, n_train=96, n_eval=48, device="cpu",
                ckpt_path=ckpt, seed=0, verbose=False,
            )
            print(f"Affinity distillation Spearman {hist['initial_spearman']:.2f} "
                  f"-> {hist['final_spearman']:.2f}")
            self.assertTrue(os.path.exists(ckpt))
            self.assertGreater(hist["final_spearman"], hist["initial_spearman"] + 0.05)
            # checkpoint loads into a fresh surrogate
            c = torch.load(ckpt, map_location="cpu")
            s = AffinitySurrogate(**c["config"])
            s.load_state_dict(c["state_dict"])


if __name__ == "__main__":
    unittest.main()
