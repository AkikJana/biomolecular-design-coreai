# Adversarial Review Report

## Challenge Summary

**Overall risk assessment**: LOW

The proposed modifications are robust to the targeted failure modes (GRPO loss degeneracy and flow-matching integration resets). However, adversarial inputs or hyperparameter settings could expose edge-case failure modes in biological coordinate space or RL stability.

## Challenges

### [Low] Challenge 1: Single-Sequence Group GRPO Degeneracy
- **Assumption challenged**: GRPO assumes a group size $G > 1$ with variance in rewards to compute standardized advantages.
- **Attack scenario**: If a user runs `run_codesign_loop` or training with `group_size = 1` or if all sequences get the same reward, `std_r` becomes 0.
- **Blast radius**: The division by `std_r` becomes `0 / (0 + 1e-8) = 0`, leading to zero advantages and no policy updates. The policy will stop learning, but it will not crash with NaNs.
- **Mitigation**: The code includes `+ 1e-8` which successfully prevents division-by-zero crashes. To fully mitigate, a check `if std_r < 1e-6: advantages = torch.zeros_like(rewards)` or enforcing `group_size >= 2` would be recommended.

### [Low] Challenge 2: Disconnected Flow-Matching Manifold Trajectories
- **Assumption challenged**: During speculative flow matching, biophysical projections are applied at every accepted and corrected step.
- **Attack scenario**: If the draft model proposes highly distorted structures, projecting them to the CA-CA bond length manifold (3.80 A) at every step might lead to jerky or physically disjoint trajectories since the vector field function is evaluated at the projected states rather than the continuous ODE states.
- **Blast radius**: Increased target model rejection rates because the draft trajectories diverge too much from target expectations.
- **Mitigation**: The implementation allows disabling biophysical constraints using `enable_biophysical=False` to recover standard flow matching trajectories if structural distortion is too high.

## Stress Test Results

- **Group reward standard deviation is zero** → `grpo_loss` computes zero advantages and zero gradient updates → **PASS** (expected behavior, stable execution)
- **Draft model matches target model perfectly** → L2 discrepancy is 0.0 → **PASS** (verified via `test_speculative_flow_matching.py` Case A)

## Unchallenged Areas

- **Full scale biological training (large sequence length)** — reason not challenged: Computational constraints of the local CPU/Sandbox environment.
