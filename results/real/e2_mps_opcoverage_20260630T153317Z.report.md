# E2 - MPS op-coverage / fallback report

- run_id: `e2_mps_opcoverage_20260630T153317Z`
- timestamp: 2026-06-30T15:33:17.019978+00:00
- device: **mps** | dtype: **fp16**
- real model ran end-to-end on MPS: **yes**
- weights: `/Users/akikjana/.boltz/boltz1_conf.ckpt`
  - weights_version: **Boltz-1 (boltz1_conf.ckpt)**
  - weights_sha256: `fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5`
- input: **SYNTHETIC feats (n_tokens=24, n_atoms=64, n_msa=8) - real model + real weights, synthetic input**
- code_sha: `e41b76714692c5f99fff7876bed9594cc1740972` | boltz_commit: `e41b76714692c5f99fff7876bed9594cc1740972`
- opm_mode (BOLTZMAC_OPM): **stock**
- hardware: arm | os: Darwin 25.6.0 (macOS-26.6-arm64-arm-64bit-Mach-O)
- torch: 2.9.1
- seed: 0
- command: `python e2_mps_opcoverage.py --fast`

## Notes & caveats

- WEIGHTS CAVEAT: the loaded checkpoint is **Boltz-1**, not Boltz-2. The op-coverage / fallback map is architecture-driven so this is still informative, but profiling real Boltz-2 requires Boltz-2 weights.
- INPUT CAVEAT: synthetic feats (correct keys/shapes/dtypes) are used because op-coverage is architecture-driven; swap in a real BoltzInferenceDataModule feats dict for numeric fidelity (no other change needed).
- PYTORCH_ENABLE_MPS_FALLBACK=1 - unsupported MPS ops fall back to CPU and emit a UserWarning, which we harvest as the authoritative fallback-op list.
- MPS CPU-fallback ops detected via warnings: ['aten::linalg_svd']

## Wall-clock by stage

> Smoke-run timings, not steady-state: the total includes one-time MPS kernel compilation / warmup (no separate warmup pass is run).

| stage | seconds | % |
|---|---:|---:|
| input_featurization | 0.6205 | 5.1% |
| trunk_msa_module | 0.5798 | 4.7% |
| trunk_pairformer | 3.3942 | 27.7% |
| distogram | 0.0077 | 0.1% |
| diffusion_sampler | 1.7894 | 14.6% |
| confidence_head | 5.8798 | 47.9% |
| **total** | **12.2714** | 100% |

## Op coverage / device-fallback table

Verdicts: **mps** = ran on MPS; **cpu_fallback** = MPS-unsupported, fell back to CPU (confirmed by PyTorch's own fallback UserWarning); **cpu_native** = only ever saw CPU tensors; **unsupported** = raised.

> `direct_mps_kernel` is informational only (does the op have a *dedicated* MPS registration). `NO` does **not** imply a fallback: structural/factory ops (view/expand/clone/_to_copy/arange/randn) run on MPS via composite or fall-through kernels. Fallback verdicts come solely from the fallback warning set.

> **Ground truth for fallbacks is PyTorch's own *unsupported-MPS* fallback warnings.** Silent host<->device scalar syncs (e.g. `.item()` / `aten::_local_scalar_dense`) emit no such warning, so they are classified **mps** and never appear here. This table therefore reports *unsupported-op* CPU fallbacks, not data-transfer / sync overhead — so the defensible claim is "the only unsupported-MPS fallback **observed** was `aten::linalg_svd`", not "everything else runs on MPS".

### Per-op rollup (all stages)

| op | calls | verdict | direct_mps_kernel | stages |
|---|---:|---|:--:|---|
| `aten::linalg_svd` | 2 | cpu_fallback | yes | diffusion_sampler |
| `aten::_to_copy` | 10971 | mps | NO | attention_pair_bias, confidence_head, diffusion_sampler, distogram, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer, uncategorized |
| `aten::view` | 5603 | mps | yes | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::linear` | 4405 | mps | yes | attention_pair_bias, confidence_head, diffusion_sampler, distogram, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer, uncategorized |
| `aten::permute` | 3629 | mps | yes | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::mul` | 3037 | mps | ? | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::unsqueeze` | 2907 | mps | NO | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::expand` | 1874 | mps | NO | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::slice` | 1687 | mps | ? | confidence_head, diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::_unsafe_view` | 1516 | mps | NO | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::add` | 1458 | mps | ? | attention_pair_bias, confidence_head, diffusion_sampler, distogram, input_featurization, trunk_msa_module, trunk_pairformer, uncategorized |
| `aten::native_layer_norm` | 1382 | mps | yes | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer, uncategorized |
| `aten::clone` | 1308 | mps | NO | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::transpose` | 1155 | mps | ? | confidence_head, diffusion_sampler, distogram, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::sigmoid` | 1062 | mps | yes | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::bmm` | 1037 | mps | yes | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::div` | 625 | mps | ? | attention_pair_bias, confidence_head, diffusion_sampler, input_featurization, trunk_msa_module, trunk_pairformer |
| `aten::rand` | 424 | mps | NO | confidence_head, trunk_msa_module, trunk_pairformer |
| `aten::ge` | 424 | mps | ? | confidence_head, trunk_msa_module, trunk_pairformer |
| `aten::add_` | 416 | mps | ? | confidence_head, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::_softmax` | 383 | mps | yes | attention_pair_bias, confidence_head, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::silu` | 282 | mps | yes | confidence_head, diffusion_sampler, input_featurization, trunk_msa_module, trunk_pairformer |
| `aten::split` | 278 | mps | ? | confidence_head, diffusion_sampler, input_featurization, triangle_multiplication |
| `aten::sub` | 238 | mps | ? | confidence_head, diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::copy_` | 211 | mps | NO | confidence_head, diffusion_sampler, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::div_` | 208 | mps | ? | confidence_head, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::new_zeros` | 208 | mps | NO | confidence_head, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::rsub` | 178 | mps | ? | attention_pair_bias, confidence_head, diffusion_sampler, trunk_msa_module |
| `aten::sum` | 55 | mps | NO | confidence_head, diffusion_sampler, input_featurization, trunk_msa_module |
| `aten::clamp` | 22 | mps | yes | confidence_head, diffusion_sampler, input_featurization, trunk_msa_module |
| `aten::select` | 21 | mps | ? | confidence_head, diffusion_sampler, input_featurization |
| `aten::relu` | 19 | mps | yes | confidence_head, diffusion_sampler, input_featurization |
| `aten::squeeze` | 19 | mps | NO | attention_pair_bias, confidence_head |
| `aten::cat` | 16 | mps | yes | confidence_head, diffusion_sampler, input_featurization, trunk_msa_module |
| `aten::eq` | 16 | mps | ? | confidence_head, diffusion_sampler, input_featurization |
| `aten::zeros` | 14 | mps | NO | confidence_head, diffusion_sampler, input_featurization |
| `aten::detach` | 14 | mps | NO | confidence_head, diffusion_sampler |
| `aten::_local_scalar_dense` | 13 | mps | yes | confidence_head, diffusion_sampler, input_featurization |
| `aten::arange` | 12 | mps | NO | confidence_head, diffusion_sampler, input_featurization |
| `aten::zeros_like` | 11 | mps | NO | confidence_head, diffusion_sampler, input_featurization |
| `aten::reciprocal` | 10 | mps | yes | confidence_head, diffusion_sampler, input_featurization |
| `aten::bitwise_and` | 10 | mps | ? | confidence_head, diffusion_sampler, input_featurization |
| `aten::ne` | 10 | mps | ? | confidence_head, diffusion_sampler, input_featurization |
| `aten::scatter_` | 9 | mps | ? | confidence_head, diffusion_sampler, input_featurization |
| `aten::where` | 9 | mps | NO | confidence_head, diffusion_sampler, input_featurization |
| `aten::pow` | 9 | mps | ? | confidence_head, diffusion_sampler |
| `aten::unbind` | 7 | mps | ? | confidence_head, diffusion_sampler |
| `aten::randn` | 7 | mps | NO | diffusion_sampler |
| `aten::sqrt` | 6 | mps | yes | diffusion_sampler |
| `aten::lt` | 6 | mps | ? | confidence_head, diffusion_sampler |
| `aten::max` | 6 | mps | yes | confidence_head |
| `aten::any` | 4 | mps | yes | confidence_head, diffusion_sampler, input_featurization |
| `aten::gt` | 4 | mps | ? | confidence_head, diffusion_sampler |
| `aten::full` | 4 | mps | NO | diffusion_sampler |
| `aten::log` | 4 | mps | yes | diffusion_sampler |
| `aten::cos` | 4 | mps | yes | diffusion_sampler |
| `aten::index_put_` | 4 | mps | NO | diffusion_sampler |
| `aten::scalar_tensor` | 3 | mps | NO | confidence_head, diffusion_sampler |
| `aten::mean` | 3 | mps | NO | diffusion_sampler |
| `aten::index` | 3 | mps | ? | confidence_head, diffusion_sampler |
| `aten::abs` | 3 | mps | NO | confidence_head, diffusion_sampler |
| `aten::eye` | 3 | mps | NO | confidence_head, diffusion_sampler |
| `aten::neg` | 2 | mps | yes | diffusion_sampler |
| `aten::stack` | 2 | mps | NO | diffusion_sampler |
| `aten::le` | 2 | mps | ? | diffusion_sampler |
| `aten::repeat` | 2 | mps | yes | diffusion_sampler |
| `aten::_linalg_det` | 2 | mps | yes | diffusion_sampler |
| `aten::_unique2` | 2 | mps | yes | confidence_head |
| `aten::norm` | 2 | mps | ? | confidence_head |
| `aten::constant_pad_nd` | 1 | mps | yes | diffusion_sampler |
| `aten::sub_` | 1 | mps | ? | diffusion_sampler |
| `aten::_cdist_forward` | 1 | mps | yes | confidence_head |
| `aten::embedding` | 1 | mps | NO | confidence_head |

### Per-stage summary

| stage | total ops | distinct ops | cpu_fallback ops |
|---|---:|---:|---|
| input_featurization | 424 | 35 | - (none observed) |
| trunk_msa_module | 890 | 29 | - (none observed) |
| trunk_pairformer | 6336 | 25 | - (none observed) |
| distogram | 6 | 4 | - (none observed) |
| diffusion_sampler | 4197 | 61 | `aten::linalg_svd` |
| confidence_head | 8032 | 58 | - (none observed) |

## Static analysis: Boltz ops with known MPS risk

| stage | op | where | risk |
|---|---|---|---|
| diffusion / SVD alignment | `aten::linalg_svd` | boltz.model.loss.diffusion.weighted_rigid_align (torch.linalg.svd, driver='gesvd' on CUDA only), called from AtomDiffusion.sample under `alignment_reverse_diff` | OBSERVED empirically: linalg_svd has no MPS kernel and falls back to CPU (PyTorch emits its unsupported-MPS fallback warning). It is hit on EVERY reverse-diffusion step of the DEFAULT (unsteered) sampler via alignment_reverse_diff - i.e. on every Boltz inference - not only in training/steered sampling. It was the ONLY unsupported-MPS fallback observed on this path (silent host<->device scalar syncs are not counted as fallbacks). |
| diffusion / SVD alignment | `aten::linalg_qr / aten::linalg_eigh` | linear-algebra helpers around rigid alignment | linalg factorizations are commonly CPU-only on MPS. |
| trunk / triangle attention | `aten::scaled_dot_product_attention / softmax / einsum` | TriangleAttentionStartingNode / EndingNode, AttentionPairBias | Generally supported on MPS, but the fused SDPA kernel coverage varies by torch version; einsum decomposes to bmm/permute which are supported. |
| trunk / triangle multiplication | `aten::einsum -> aten::bmm` | TriangleMultiplicationOutgoing / Incoming | Supported on MPS (bmm/mul/sigmoid). Watch for fp16 autocast edge cases. |
| diffusion sampler | `aten::randn / aten::native_layer_norm / aten::index_add` | AtomDiffusion.sample noise + atom-attention scatter/gather | index/scatter ops occasionally fall back; randn supported on MPS. |
| input featurization | `aten::one_hot / aten::cdist` | encoders / relative position encoding | cdist and some indexing ops have had MPS gaps across torch versions. |
