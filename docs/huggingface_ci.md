# Publishing to Hugging Face from GitHub

Two things live on the Hub. Only one of them can be published by CI, and the
reason is worth stating rather than discovering.

| | Hub repo | published by |
| :-- | :-- | :-- |
| Static Space | `AkikJana/scramble-control-scorer` | GitHub Actions, on merge to main |
| Dataset | `AkikJana/scramble-control-panels` | you, from a machine with `artifacts/` |

## The Space syncs automatically

`.github/workflows/hf-space.yml` runs on any merge to `main` that touches
`hf_space/**`, and uploads the folder with `src/push_hf_space.py`.

It needs one repository secret, which you have to add yourself:

**Settings → Secrets and variables → Actions → New repository secret**

- Name: `HF_TOKEN`
- Value: a token from <https://huggingface.co/settings/tokens>

Make it a **fine-grained** token with write access to *that one Space* and
nothing else. A classic write token can push to every repo on the account,
including the dataset, and this workflow has no business touching the dataset.

Nothing else is required. The workflow prints what it is about to upload before
uploading it, so the run log shows the diff-by-file even though the Hub commit
does not.

### Why it does not run on pull requests

Secrets are withheld from fork PRs, so a fork could not publish in any case. A
same-repo PR *would* have the secret, which would mean any open branch could
change the public site before review. Publishing follows `main`.

### Why it cannot create the Space

`push_hf_space.py` refuses to create a repo unless `--create` is passed, and the
workflow never passes it. An unattended job that can create repos is one that can
publish to a mistyped name, with visibility set by whatever the default was. The
Space already exists; CI only updates it.

## The dataset does not, and should not

`build_hf_dataset.py` reads four things out of `artifacts/`:

    artifacts/pdb_binders_b2_n22/     panel inputs and MSA cache
    artifacts/<arm>.json              per-fold scores for all 16 arms
    artifacts/posebusters.json        validity run

`artifacts/` is gitignored — it is tens of gigabytes of folds. A CI runner
checking out this repository does not have it and cannot reconstruct it without
re-folding, which is the entire cost of the project.

The build already fails loudly on a missing arm rather than writing a short file:

```python
if not p.exists():
    raise SystemExit(f"missing artifact: {p}")
```

That is the property that makes the current arrangement safe. Wiring the dataset
into CI "for completeness" would either fail on every run, or — if someone later
made the missing arms optional to get the job green — silently publish a dataset
with arms missing and no indication which. A truncated dataset that looks whole
is worse than no automation.

So the dataset is pushed by hand, from a machine that has the folds:

```bash
python src/build_hf_dataset.py && python src/push_hf_dataset.py --dry-run
```

Read the file list, then drop `--dry-run`.

## Checking it worked

```bash
gh run list --workflow "Sync HF Space" --limit 3
```
