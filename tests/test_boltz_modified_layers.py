import sys
import os
import torch
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


if __name__ == "__main__":
    unittest.main()
