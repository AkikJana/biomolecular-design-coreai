# Configuration And Seeds

`config.yaml` is the base reproducibility template. `seed.txt` is the canonical plain-text seed convention for experiments that do not parse YAML directly.

Experiment runners should read the seed from the run folder first:

1. `experiments/<run-id>/seed.txt`
2. `experiments/<run-id>/config.yaml`
3. `configs/seed.txt`
4. `configs/config.yaml`

If more than one seed source is present for a run, the values must match or the run should fail before producing results.
