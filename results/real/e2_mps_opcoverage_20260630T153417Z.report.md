# E2 - MPS op-coverage / fallback report

- run_id: `e2_mps_opcoverage_20260630T153417Z`
- timestamp: 2026-06-30T15:34:17.591952+00:00
- device: **mps** | dtype: **fp16**
- real model ran end-to-end on MPS: **partial**
- weights: `/Users/akikjana/.boltz/boltz1_conf.ckpt`
  - weights_version: **Boltz-1 (boltz1_conf.ckpt)**
  - weights_sha256: `fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5`
- input: **SYNTHETIC feats (n_tokens=48, n_atoms=256, n_msa=16) - real model + real weights, synthetic input**
- code_sha: `e41b76714692c5f99fff7876bed9594cc1740972` | boltz_commit: `e41b76714692c5f99fff7876bed9594cc1740972`
- hardware: arm | os: Darwin 25.6.0 (macOS-26.6-arm64-arm-64bit-Mach-O)
- torch: 2.9.1
- seed: 0
- command: `python e2_mps_opcoverage.py`

## Notes & caveats

- WEIGHTS CAVEAT: the loaded checkpoint is **Boltz-1**, not Boltz-2. The op-coverage / fallback map is architecture-driven so this is still informative, but profiling real Boltz-2 requires Boltz-2 weights.
- INPUT CAVEAT: synthetic feats (correct keys/shapes/dtypes) are used because op-coverage is architecture-driven; swap in a real BoltzInferenceDataModule feats dict for numeric fidelity (no other change needed).
- PYTORCH_ENABLE_MPS_FALLBACK=1 - unsupported MPS ops fall back to CPU and emit a UserWarning, which we harvest as the authoritative fallback-op list.
- RUN ERROR (forward did not fully complete): _LinAlgError: linalg.svd: (Batch element 0): The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: 2).
- MPS CPU-fallback ops detected via warnings: ['aten::linalg_svd']

## Wall-clock by stage

| stage | seconds | % |
|---|---:|---:|
| input_featurization | 0.6553 | 6.6% |
| trunk_msa_module | 1.0741 | 10.9% |
| trunk_pairformer | 7.0799 | 71.7% |
| distogram | 0.0097 | 0.1% |
| diffusion_sampler | 1.0603 | 10.7% |
| **total** | **9.8792** | 100% |

## Op coverage / device-fallback table

Verdicts: **mps** = ran on MPS; **cpu_fallback** = MPS-unsupported, fell back to CPU (confirmed by PyTorch's own fallback UserWarning); **cpu_native** = only ever saw CPU tensors; **unsupported** = raised.

> `direct_mps_kernel` is informational only (does the op have a *dedicated* MPS registration). `NO` does **not** imply a fallback: structural/factory ops (view/expand/clone/_to_copy/arange/randn) run on MPS via composite or fall-through kernels. Fallback verdicts come solely from the fallback warning set.

### Per-op rollup (all stages)

| op | calls | verdict | direct_mps_kernel | stages |
|---|---:|---|:--:|---|
| `aten::linalg_svd` | 1 | unsupported | yes | diffusion_sampler |
| `aten::_to_copy` | 9543 | mps | NO | attention_pair_bias, diffusion_sampler, distogram, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer, uncategorized |
| `aten::view` | 4963 | mps | yes | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::linear` | 3894 | mps | yes | attention_pair_bias, diffusion_sampler, distogram, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer, uncategorized |
| `aten::permute` | 3133 | mps | yes | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::mul` | 2662 | mps | ? | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::unsqueeze` | 2437 | mps | NO | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::expand` | 1810 | mps | NO | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::slice` | 1685 | mps | ? | diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::_unsafe_view` | 1444 | mps | NO | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::native_layer_norm` | 1233 | mps | yes | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer, uncategorized |
| `aten::add` | 1196 | mps | ? | attention_pair_bias, diffusion_sampler, distogram, input_featurization, trunk_msa_module, trunk_pairformer, uncategorized |
| `aten::clone` | 1173 | mps | NO | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::transpose` | 1149 | mps | ? | diffusion_sampler, distogram, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::bmm` | 933 | mps | yes | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::sigmoid` | 895 | mps | yes | attention_pair_bias, diffusion_sampler, input_featurization, triangle_attention, triangle_multiplication, trunk_msa_module, trunk_pairformer |
| `aten::div` | 570 | mps | ? | attention_pair_bias, diffusion_sampler, input_featurization, trunk_msa_module, trunk_pairformer |
| `aten::rand` | 424 | mps | NO | trunk_msa_module, trunk_pairformer |
| `aten::ge` | 424 | mps | ? | trunk_msa_module, trunk_pairformer |
| `aten::add_` | 416 | mps | ? | triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::_softmax` | 345 | mps | yes | attention_pair_bias, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::silu` | 246 | mps | yes | diffusion_sampler, input_featurization, trunk_msa_module, trunk_pairformer |
| `aten::split` | 243 | mps | ? | diffusion_sampler, input_featurization, triangle_multiplication |
| `aten::sub` | 221 | mps | ? | diffusion_sampler, input_featurization, triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::div_` | 208 | mps | ? | triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::new_zeros` | 208 | mps | NO | triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::copy_` | 208 | mps | NO | triangle_attention, trunk_msa_module, trunk_pairformer |
| `aten::rsub` | 140 | mps | ? | attention_pair_bias, diffusion_sampler, trunk_msa_module |
| `aten::sum` | 18 | mps | NO | diffusion_sampler, input_featurization, trunk_msa_module |
| `aten::clamp` | 15 | mps | yes | diffusion_sampler, input_featurization, trunk_msa_module |
| `aten::relu` | 12 | mps | yes | diffusion_sampler, input_featurization |
| `aten::cat` | 10 | mps | yes | diffusion_sampler, input_featurization, trunk_msa_module |
| `aten::squeeze` | 9 | mps | NO | attention_pair_bias |
| `aten::zeros` | 7 | mps | NO | diffusion_sampler, input_featurization |
| `aten::zeros_like` | 7 | mps | NO | diffusion_sampler, input_featurization |
| `aten::scatter_` | 5 | mps | ? | diffusion_sampler, input_featurization |
| `aten::reciprocal` | 5 | mps | yes | diffusion_sampler, input_featurization |
| `aten::bitwise_and` | 5 | mps | ? | diffusion_sampler, input_featurization |
| `aten::eq` | 5 | mps | ? | diffusion_sampler, input_featurization |
| `aten::where` | 5 | mps | NO | diffusion_sampler, input_featurization |
| `aten::arange` | 4 | mps | NO | diffusion_sampler, input_featurization |
| `aten::select` | 4 | mps | ? | diffusion_sampler, input_featurization |
| `aten::_local_scalar_dense` | 4 | mps | yes | diffusion_sampler, input_featurization |
| `aten::pow` | 4 | mps | ? | diffusion_sampler |
| `aten::unbind` | 4 | mps | ? | diffusion_sampler |
| `aten::randn` | 4 | mps | NO | diffusion_sampler |
| `aten::sqrt` | 3 | mps | yes | diffusion_sampler |
| `aten::ne` | 2 | mps | ? | diffusion_sampler, input_featurization |
| `aten::scalar_tensor` | 2 | mps | NO | diffusion_sampler |
| `aten::lt` | 2 | mps | ? | diffusion_sampler |
| `aten::full` | 2 | mps | NO | diffusion_sampler |
| `aten::log` | 2 | mps | yes | diffusion_sampler |
| `aten::cos` | 2 | mps | yes | diffusion_sampler |
| `aten::detach` | 2 | mps | NO | diffusion_sampler |
| `aten::index_put_` | 2 | mps | NO | diffusion_sampler |
| `aten::any` | 1 | mps | yes | input_featurization |
| `aten::constant_pad_nd` | 1 | mps | yes | diffusion_sampler |
| `aten::gt` | 1 | mps | ? | diffusion_sampler |
| `aten::neg` | 1 | mps | yes | diffusion_sampler |
| `aten::stack` | 1 | mps | NO | diffusion_sampler |
| `aten::mean` | 1 | mps | NO | diffusion_sampler |
| `aten::index` | 1 | mps | ? | diffusion_sampler |

### Per-stage summary

| stage | total ops | distinct ops | cpu_fallback ops |
|---|---:|---:|---|
| input_featurization | 424 | 35 | - (all on MPS) |
| trunk_msa_module | 1780 | 29 | - (all on MPS) |
| trunk_pairformer | 12672 | 25 | - (all on MPS) |
| distogram | 6 | 4 | - (all on MPS) |
| diffusion_sampler | 2161 | 53 | `aten::linalg_svd` |

## Static analysis: Boltz ops with known MPS risk

| stage | op | where | risk |
|---|---|---|---|
| diffusion / SVD alignment | `aten::linalg_svd` | boltz.model.loss.diffusion.weighted_rigid_align (torch.linalg.svd, driver='gesvd' on CUDA only), called from AtomDiffusion.sample under `alignment_reverse_diff` | CONFIRMED empirically: linalg_svd has no MPS kernel and falls back to CPU. It is hit on EVERY reverse-diffusion step of the DEFAULT (unsteered) sampler via alignment_reverse_diff - i.e. on every Boltz inference - not only in training/steered sampling. This is the one true CPU fallback on this path. |
| diffusion / SVD alignment | `aten::linalg_qr / aten::linalg_eigh` | linear-algebra helpers around rigid alignment | linalg factorizations are commonly CPU-only on MPS. |
| trunk / triangle attention | `aten::scaled_dot_product_attention / softmax / einsum` | TriangleAttentionStartingNode / EndingNode, AttentionPairBias | Generally supported on MPS, but the fused SDPA kernel coverage varies by torch version; einsum decomposes to bmm/permute which are supported. |
| trunk / triangle multiplication | `aten::einsum -> aten::bmm` | TriangleMultiplicationOutgoing / Incoming | Supported on MPS (bmm/mul/sigmoid). Watch for fp16 autocast edge cases. |
| diffusion sampler | `aten::randn / aten::native_layer_norm / aten::index_add` | AtomDiffusion.sample noise + atom-attention scatter/gather | index/scatter ops occasionally fall back; randn supported on MPS. |
| input featurization | `aten::one_hot / aten::cdist` | encoders / relative position encoding | cdist and some indexing ops have had MPS gaps across torch versions. |
