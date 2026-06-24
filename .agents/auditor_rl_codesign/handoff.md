# Forensic Audit Report: RL Co-Design Loop & Speculative Flow Matching

**Work Product**: Biomolecular Design Repository (RL Co-Design files)  
**Profile**: General Project  
**Verdict**: VIOLATION DETECTED  

---

## 1. Observation

I inspected the following files in the workspace:
* **`src/train_preference_alignment.py`**
* **`src/speculative_flow_matching.py`**
* **`src/agentic_design_loop.py`**
* **`tests/test_agentic_design_loop.py`**

Specifically, I observed the following implementation details:

### Observation A: Degenerate GRPO Loss & Zero KL Gradient
In `src/train_preference_alignment.py` (lines 474–476):
```python
                with torch.no_grad():
                    old_logits = policy_model(tokens)
                    old_logps = get_sequence_logps(old_logits, tokens, length_normalize=True).detach()
```
And in `src/agentic_design_loop.py` (lines 114–116):
```python
        # Compute old log probabilities
        with torch.no_grad():
            old_logits = policy(encoded_tokens)
            old_logps = get_sequence_logps(old_logits, encoded_tokens, length_normalize=True).detach()
```
In both files, the active policy network (`policy_model`/`policy`) is evaluated to compute `old_logps` at the exact same step, on the same inputs, and using the same weights as the active forward pass:
```python
        # Forward pass on active policy
        policy_logits = policy(encoded_tokens)
        policy_logps = get_sequence_logps(policy_logits, encoded_tokens, length_normalize=True)
```
These log probabilities are then passed into the GRPO loss function `grpo_loss` (defined in `src/train_preference_alignment.py` at lines 241–279):
```python
    # Policy ratio calculation
    ratios = torch.exp(policy_logps - old_logps)
    ...
    # Reference-free KL divergence penalty:
    log_ratio = old_logps - policy_logps
    kl = torch.exp(log_ratio) - log_ratio - 1
```

### Observation B: Overwritten/Discarded Intermediate Speculative Step Corrections
In `src/speculative_flow_matching.py` (lines 214–240), the loop iterates through speculative steps:
```python
            for k in range(actual_k):
                x_k = draft_x[k]
                t_k = draft_t[k]
                t_k_tensor = torch.full((batch_size,), t_k, device=device, dtype=dtype)
                
                # Retrieve target vector field evaluated at x_k
                v_target = v_target_steps[k]
                
                # Re-evaluate draft vector field at x_k to check divergence
                v_draft = self.draft_vf_fn(x_k, t_k_tensor, **extra_args)
                
                # Measure L2 difference between vector fields (normalized by magnitude)
                diff = torch.norm(v_target - v_draft, p=2, dim=-1) / (torch.norm(v_target, p=2, dim=-1) + 1e-8)
                mean_diff = diff.mean().item()
                
                if mean_diff <= self.tolerance:
                    # Accept step: update state using the target vector field (semi-correction)
                    curr_verified_x = x_k + v_target * dt
                    accepted_k += 1
                    total_drafts_accepted += 1
                else:
                    # Reject step: correct the current step using the target model's trajectory
                    curr_verified_x = x_k + v_target * dt
                    break
```

---

## 2. Logic Chain

### Logic Chain A: Degenerate GRPO Loss
1. Because `policy_logits` and `old_logits` are evaluated at the exact same active policy weights, `policy_logps` and `old_logps` are mathematically identical in the forward pass of every step.
2. In `grpo_loss`, `log_ratio = old_logps - policy_logps` evaluates to `0.0`.
3. Consequently, `kl = torch.exp(log_ratio) - log_ratio - 1` evaluates to `torch.exp(0.0) - 0.0 - 1 = 0.0`. Its derivative/gradient with respect to the active `policy_logps` is $-\exp(0.0) + 1 = 0.0$. Thus, the regularizing KL divergence penalty is completely non-functional.
4. The policy ratio `ratios = torch.exp(0.0)` evaluates to `1.0`. Since `clip_eps` is `0.2` (meaning the clipping range is `[0.8, 1.2]`), the clipped surrogate loss `clip_loss` reduces to `advantages`.
5. The total loss `loss = -(clip_loss - beta * kl).mean()` simplifies to `-advantages.mean()`.
6. Since advantages are standardized to have zero-mean across the group (`advantages = (rewards - mean_r) / std_r`), their mean is mathematically guaranteed to be exactly `0.0`.
7. Therefore, the logged GRPO training loss evaluates to exactly `0.0000` at every training iteration, rendering the reported training metrics degenerate. A correct GRPO implementation requires anchoring regularizations against a frozen separate reference model (like `ref_model` in DPO) or updating parameters across multiple inner epochs.

### Logic Chain B: Discarded Speculative Step Corrections
1. In `src/speculative_flow_matching.py` (lines 230–237), when a draft step is verified (either accepted or rejected/corrected), the variable `curr_verified_x` is set to `x_k + v_target * dt`.
2. `x_k = draft_x[k]` is the candidate state generated purely by the draft model during the sequential draft rollout (lines 165–181). It does *not* incorporate any target-model corrections from step `0` to step `k-1`.
3. In each iteration of the loop over `k`, `curr_verified_x` is completely overwritten.
4. As a result, all target-model corrections for intermediate steps `0` to `actual_k - 2` are completely discarded. Only the correction of the very last step (`actual_k - 1` or the step where it broke) is preserved in `x = curr_verified_x` at line 240.
5. This means that the speculative sampler integrates along the uncorrected draft trajectory for almost all steps, failing to correct the accumulation of drift from the draft model and violating the mathematical correctness of speculative ODE solvers.

---

## 3. Caveats

* Testing in this environment was constrained to static and manual code analysis because command execution permissions timed out (preventing dynamic runtime test execution). However, the mathematical behaviors of the PyTorch operations are clear from source inspection.
* Mock models are used for vector fields in demonstrations, which is acceptable given the lack of production Boltz weights in the repository.

---

## 4. Conclusion

The work product contains two significant mathematical and logic violations in PyTorch calculations:
1. The KL divergence regularizer is evaluated against the active policy itself in the same step, resulting in a degenerate KL divergence of `0.0` and a constant GRPO loss value of exactly `0.0000` at every step.
2. The speculative sampler discards intermediate target-model step corrections, resulting in an uncorrected drift from the draft model.

Therefore, the verdict is **VIOLATION DETECTED**, and the work product must be rejected.

---

## 5. Verification Method

To dynamically verify the mathematical zero-mean loss and degenerate metrics, execute the co-design tests:
1. Run the test script inside the project's virtual environment:
   ```bash
   .venv/bin/python -m pytest tests/test_agentic_design_loop.py -v
   ```
2. Verify in the training output of `run_codesign_loop` that the printed loss and KL divergence values are always exactly `0.0000` for all iterations:
   ```
   Iteration 01 | Mean Reward: ... | Loss: 0.0000 | Avg KL: 0.0000
   Iteration 02 | Mean Reward: ... | Loss: 0.0000 | Avg KL: 0.0000
   Iteration 03 | Mean Reward: ... | Loss: 0.0000 | Avg KL: 0.0000
   ```
