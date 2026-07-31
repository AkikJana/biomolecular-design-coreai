# Frozen Benchmark Contract

This directory is reserved for benchmark inputs and labels that have been frozen for reproducible thesis evaluation.

Once a benchmark set is frozen:

- input files, labels, complex ids, and metric definitions are immutable;
- corrections require a new frozen-set directory or version, not in-place edits;
- every run against the frozen set must record the manifest provenance needed to reproduce the run;
- placeholder or example manifests must be clearly marked and must not be cited as completed benchmark results.

The `manifest.json` file is a template for the provenance record expected beside a frozen set. Replace placeholder values only when a benchmark set is actually frozen.
