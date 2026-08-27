# What the repository checks on its own

GitHub cannot do research. What it can do is make the failures this project has
actually had impossible to reintroduce quietly. Every one of them was a silent
substitution or a stale number — never a missing idea — so that is what the
automation targets.

| failure that actually happened | what now catches it | where |
| :-- | :-- | :-- |
| `pip install boltz` fetched a different codebase | fork guard, three places | Dockerfile build, Docker CI, `provision.sh` |
| `pair_000` named a different molecule per run | duplicate name is fatal | `build_hf_dataset.py` |
| run-to-run SD survived only as a literal | written to an artifact | `replicate_noise.py` |
| a number changed in one place and not the other | claims lock | `ci.yml` |
| the Space published from an unreviewed branch | push-to-main only | `hf-space.yml` |

## The claims lock

`verify_claims.py` recomputes 25 load-bearing figures from `artifacts/`. That
directory is ~200 GB of folds and will never be in a clone, so CI cannot run the
recomputation.

The lock splits it in two:

```bash
python src/verify_claims.py --emit results/claims_lock.json
```

Run on a machine that has the folds. It recomputes everything and freezes the
result into a committed file.

```bash
python src/verify_claims.py --check results/claims_lock.json
```

Runs anywhere, needs no artifacts, and is wired into CI. It holds the report to
the lock.

**What this catches:** editing a figure in the report without re-deriving it, and
leaving a sentence describing data that has since been replaced. The stale
43-page PDF, the "median 41 fragments" that was really 40.5, and the sentence
still saying 8 targets after the run covered 10 were all this class, and all
found by hand.

**What it does not catch:** the artifacts changing while the lock stays put. Only
`--emit` on a machine with folds can tell you that. After any run that changes a
load-bearing number, re-emit and commit the diff — the diff is the point, since
it makes the change visible in review rather than silent.

## Deliberately not set up

**Dependabot.** This repository's most expensive error was a dependency
substitution: `pip install boltz` quietly resolved to a different codebase and
cost the attribution of a published number. A bot that opens version bumps is the
same mechanism on a schedule. The pins in `Dockerfile` and `requirements.txt` are
load-bearing scientific parameters, not maintenance debt, and they should move
when someone decides to re-measure — not when a release lands upstream.

If it is ever enabled, it must be scoped to exclude `torch` and `boltz`, and the
Docker CI fork guard must stay green as a required check.

**Auto-merge on green.** CI here proves the report is internally consistent and
the image builds. It cannot tell whether a claim is *true*. Nothing in this
repository should merge without someone reading it.
