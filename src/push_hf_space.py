"""Push hf_space/ to the Hub as a static Space.

Mirrors push_hf_dataset.py: building is local, pushing is outward-facing, and the
two stay separate so you can look at what you are about to publish.

The difference is that this one runs unattended in CI, which changes what the
safe default is. push_hf_dataset.py defaults to private because a human is there
to flip it. Here nobody is watching, so the rule is inverted: this refuses to
CREATE anything. The Space must already exist, and --create is required to make
one. A workflow that can conjure repos is a workflow that can publish to the
wrong name, with visibility decided by whatever the default happened to be.

Auth comes from HF_TOKEN in the environment when set, which is how CI supplies
it; otherwise from your local `huggingface-cli login`.

Usage:
    python src/push_hf_space.py --dry-run
    python src/push_hf_space.py                    # push to an existing Space
    python src/push_hf_space.py --create --public  # first time, from a terminal
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE = REPO_ROOT / "hf_space"

# A static Space is served from these; without them the push succeeds and the
# page is blank, which is a worse outcome than a failed push.
REQUIRED = ["README.md", "index.html"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="scramble-control-scorer")
    ap.add_argument("--create", action="store_true",
                    help="allow creating the Space if it does not exist")
    ap.add_argument("--public", action="store_true",
                    help="only meaningful with --create")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    missing = [f for f in REQUIRED if not (SPACE / f).exists()]
    if missing:
        raise SystemExit(f"hf_space/ is missing {', '.join(missing)}")

    # The front matter is what tells the Hub this is a static Space at all. A
    # README without it renders as a model card and the site never loads.
    head = (SPACE / "README.md").read_text()[:400]
    if "sdk: static" not in head:
        raise SystemExit("hf_space/README.md front matter does not say `sdk: static`")

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token else HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as e:
        raise SystemExit(
            "not authenticated. Set HF_TOKEN, or run `huggingface-cli login` in "
            f"your own terminal.\n  ({str(e)[:100]})")

    repo_id = f"{user}/{args.name}"
    files = sorted(p for p in SPACE.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files) / 1e6
    print(f"  repo  : {repo_id}")
    print(f"  auth  : {'HF_TOKEN' if token else 'local login'} (as {user})")
    print(f"  files : {len(files)} ({size:.2f} MB)")
    for p in files:
        print(f"          {p.relative_to(SPACE)}")

    try:
        info = api.repo_info(repo_id, repo_type="space")
        print(f"  exists: yes (private={info.private})")
    except RepositoryNotFoundError:
        print("  exists: no")
        if not args.create:
            raise SystemExit(
                f"{repo_id} does not exist. Re-run with --create --public from a "
                "terminal to make it; CI is not allowed to create repos.")

    if args.dry_run:
        print("\n  dry run -- nothing pushed")
        return

    if args.create:
        api.create_repo(repo_id, repo_type="space", space_sdk="static",
                        private=not args.public, exist_ok=True)

    api.upload_folder(folder_path=str(SPACE), repo_id=repo_id, repo_type="space",
                      commit_message=os.environ.get(
                          "HF_COMMIT_MESSAGE", "Sync hf_space/ from GitHub"))
    print(f"\n  pushed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
