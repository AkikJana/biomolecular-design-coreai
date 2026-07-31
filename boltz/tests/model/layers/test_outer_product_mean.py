import os

import pytorch_lightning
import torch
import torch.nn as nn

import unittest

from boltz.model.layers.outer_product_mean import (
    OuterProductMean,
    OuterProductMeanStock,
    select_opm_cls,
)


class OuterProductMeanTest(unittest.TestCase):
    def setUp(self):
        self.c_in = 32
        self.c_hidden = 16
        self.c_out = 64

        torch.set_grad_enabled(False)
        pytorch_lightning.seed_everything(1100)
        self.layer = OuterProductMean(self.c_in, self.c_hidden, self.c_out)

        # Initialize layer
        for name, param in self.layer.named_parameters():
            nn.init.normal_(param, mean=1.0, std=1.0)

        # Set to eval mode
        self.layer.eval()

    def test_chunk(self):
        chunk_sizes = [16, 33, 64, 83, 100]
        B, S, N = 1, 49, 84
        m = torch.randn(size=(B, S, N, self.c_in))
        mask = torch.randint(low=0, high=1, size=(B, S, N))

        with torch.no_grad():
            exp_output = self.layer(m=m, mask=mask)
            for chunk_size in chunk_sizes:
                with self.subTest(chunk_size=chunk_size):
                    act_output = self.layer(m=m, mask=mask, chunk_size=chunk_size)
                    assert torch.allclose(exp_output, act_output, atol=1e-8)


def _locate_stock_checkpoint():
    """Return the local path to stock Boltz weights, or None if absent.

    Looks in the standard ``~/.boltz`` cache (see ``boltz.main``) for a stock
    checkpoint. Never downloads anything.
    """
    cache = os.path.expanduser("~/.boltz")
    # Stock checkpoints shipped by upstream, newest first.
    for name in ("boltz1_conf.ckpt", "boltz2_conf.ckpt"):
        checkpoint = os.path.join(cache, name)
        if os.path.exists(checkpoint):
            return checkpoint
    return None


# Resolved at import so the heavy state_dict-load test can be skipped (rather
# than executed) when stock weights are not present locally.
_STOCK_CHECKPOINT = _locate_stock_checkpoint()


class StockOuterProductMeanTest(unittest.TestCase):
    """Thin checks for the default (``stock``) OPM path."""

    def test_stock_forward_finite_shape(self):
        """Stock OPM is the default and its forward is finite + correctly shaped.

        Always runs (no weights needed) so reviewers get a fast signal.
        """
        # The default toggle must resolve to the full-rank stock implementation.
        assert OuterProductMean is OuterProductMeanStock
        assert select_opm_cls("stock") is OuterProductMeanStock

        c_in, c_hidden, c_out = 32, 16, 64
        B, S, N = 2, 7, 11

        pytorch_lightning.seed_everything(0)
        layer = OuterProductMeanStock(c_in, c_hidden, c_out).eval()

        m = torch.randn(B, S, N, c_in)
        mask = torch.randint(low=0, high=2, size=(B, S, N))

        with torch.no_grad():
            out = layer(m=m, mask=mask)

        assert tuple(out.shape) == (B, N, N, c_out)
        assert torch.isfinite(out).all()

    @unittest.skipUnless(
        _STOCK_CHECKPOINT is not None,
        "stock weights not found under ~/.boltz; skipping the heavy "
        "state_dict-load test (no download performed).",
    )
    def test_stock_state_dict_load(self):
        """A real stock checkpoint's OPM weights load into a stock OPM.

        Heavy (loads the full checkpoint), so it is gated on the weights being
        present locally; never downloads them.
        """
        ckpt = torch.load(_STOCK_CHECKPOINT, map_location="cpu", weights_only=False)
        full_sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # Pull one OuterProductMean instance's parameters out of the checkpoint.
        marker = ".outer_product_mean."
        prefixes = sorted(
            {k[: k.index(marker) + len(marker)] for k in full_sd if marker in k}
        )
        assert prefixes, "no outer_product_mean weights in checkpoint"
        prefix = prefixes[0]
        sub_sd = {
            k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)
        }

        # Infer dims from the checkpoint tensors so this stays robust.
        ckpt_c_hidden, ckpt_c_in = sub_sd["proj_a.weight"].shape
        ckpt_c_out = sub_sd["proj_o.weight"].shape[0]
        stock = OuterProductMeanStock(ckpt_c_in, ckpt_c_hidden, ckpt_c_out)

        missing, unexpected = stock.load_state_dict(sub_sd, strict=True)
        assert list(missing) == []
        assert list(unexpected) == []
