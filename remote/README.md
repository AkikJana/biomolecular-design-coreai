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
ssh -p $GPU_PORT $GPU_HOST 'cd BiomolecularDesign && setsid nohup bash remote/queue.sh >> queue.log 2>&1 </dev/null & disown'
remote/sync.sh down
```

`setsid` matters. Plain `nohup` inside an `ssh` command is not enough: when the
ssh session closes, the remote shell dies and its process group is signalled,
taking the run with it. That killed a run silently at 108 of 132 folds -- the log
simply stopped after a completed batch, with no error and no OOM. `setsid` puts
the run in its own session so it survives the disconnect.

## Measured, on a RunPod RTX 4090

**26 seconds per fold** at full settings (200 sampling steps, 3 recycling passes,
undiminished MSA), with MSAs served from the synced cache. 354 folds is about
2.6 GPU-hours, roughly $1 at $0.40/hr.

For scale: the same complex takes minutes on Apple Silicon at *reduced* settings,
and Chai-1 on CPU took 2h47m for one 66-residue complex.

## Notes

**Spot instances are fine.** The runners skip folds already scored, so re-running
`queue.sh` after an interruption resumes. That resume logic exists because of
local interruptions, and it makes the cheapest tier safe.

**Pull scores, not structures.** `sync.sh down` takes JSON and CSV only.
Structures are what exhausted the local disk before and cost this project its
converged models. Use `down --all` deliberately, and only if something needs
coordinates — PoseBusters on converged structures is the one good reason.

**Provisioning reuses the image's torch** when it is CUDA-enabled, because a
pip-resolved torch frequently mismatches the host driver. Installing Boltz's
optional CUDA kernels unpinned dragged torch 2.13 (CUDA 13) over 2.8.0+cu128 and
broke torch against a 12.8 driver, so `WITH_KERNELS=1` installs them `--no-deps`
and reverts if torch moves.

**Kernels are off by default.** Boltz's fused triangle attention needs
`cuequivariance_ops_torch`, which is not a dependency of this project because the
MPS path never uses it. On CUDA it fails at *call* time rather than import time,
so static checks all pass and every fold still dies. `--no_kernels` uses the
PyTorch implementation instead, and 26 s/fold is fast enough that the kernels are
not worth the dependency risk.

**Provisioning ends with a smoke fold.** Every static check passed on a box that
could not fold anything; thirty seconds of actually folding is the only check
that meant something.
