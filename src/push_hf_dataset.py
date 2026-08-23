"""Push the built dataset to the Hub.

Separate from build_hf_dataset.py on purpose: building is local and repeatable,
pushing is outward-facing and is not. Run the build first, look at what it wrote,
then run this.

Defaults to a PRIVATE repo. Going public is one click in the Hub UI and cannot
be undone by deleting the repo afterwards, so the default is the recoverable one.

Usage:
    huggingface-cli login          # interactive, run it yourself
    python src/push_hf_dataset.py [--public] [--name scramble-control-panels]
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "hf_dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="scramble-control-panels")
    ap.add_argument("--public", action="store_true",
                    help="create the repo public; default is private")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (DATA / "README.md").exists():
        raise SystemExit(f"no dataset at {DATA} -- run build_hf_dataset.py first")

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as e:
        raise SystemExit(
            "not authenticated. Run `huggingface-cli login` in your own "
            f"terminal first.\n  ({str(e)[:100]})")

    repo_id = f"{user}/{args.name}"
    files = sorted(p for p in DATA.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"  repo    : {repo_id}")
    print(f"  private : {not args.public}")
    print(f"  files   : {len(files)} ({size:.2f} MB)")
    for p in files:
        print(f"            {p.relative_to(DATA)}")
    if args.dry_run:
        print("\n  dry run -- nothing pushed")
        return

    api.create_repo(repo_id, repo_type="dataset", private=not args.public,
                    exist_ok=True)
    api.upload_folder(folder_path=str(DATA), repo_id=repo_id,
                      repo_type="dataset",
                      commit_message="Scramble-control panels, scores and PoseBusters run")
    print(f"\n  pushed: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
