# Experiment Run Template

Copy this directory to `experiments/<run-id>/` for each run.

The run folder records the exact config, seed, metrics, and timings for one experiment. Keep real benchmark outputs under `results/real/` only when they are backed by frozen benchmark provenance. Keep generated, exploratory, placeholder, or unknown-provenance outputs under `results/synthetic/`.

Do not add experiment scripts to this template. Scripts belong in the project source or orchestration layer owned by the relevant implementation task.
