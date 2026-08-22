# Running the folds on a rented GPU

354 folds are outstanding: 222 for the 37 new receptors, and 132 for a second
model-family arm. Neither is reachable on Apple Silicon — Chai-1 hits a Metal
limitation in the pair-representation matmul, and CPU took 2h47m for one
66-residue complex at the cheapest settings.

The complexes are small (59–151 residues, median ~105), so a 24 GB consumer card
is enough. An A100 is roughly 4× the price for capacity this workload will not use.

## Order of operations

```bash
export GPU_HOST=root@1.2.3.4 GPU_PORT=22222   # from your provider
remote/sync.sh up
ssh -p $GPU_PORT $GPU_HOST 'bash BiomolecularDesign/remote/provision.sh'
ssh -p $GPU_PORT $GPU_HOST 'bash BiomolecularDesign/remote/bench.sh'
```

**Read the bench output before starting the full run.** It prints seconds per
fold and projects the total; every cost estimate before that number is a guess.

```bash
ssh -p $GPU_PORT $GPU_HOST 'cd BiomolecularDesign && nohup bash remote/queue.sh > queue.log 2>&1 &'
remote/sync.sh down
```

## Notes

**Spot instances are fine.** The runners skip folds already scored, so re-running
`queue.sh` after an interruption resumes. That resume logic exists because of
local interruptions, and it makes the cheapest tier safe.

**Pull scores, not structures.** `sync.sh down` takes JSON and CSV only.
Structures are what exhausted the local disk before and cost this project its
converged models. Use `down --all` deliberately, and only if something needs
coordinates — PoseBusters on converged structures is the one good reason.

**Provisioning reuses the image's torch** when it is CUDA-enabled, because a
pip-resolved torch frequently mismatches the host driver.
