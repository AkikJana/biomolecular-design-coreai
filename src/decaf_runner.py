"""Run the DeCAF fork of Boltz under PyTorch 2.6+.

DeCAF ships an inference-only fork of Boltz. Two things stop it running here
unmodified, and both are already known problems in this project:

1. **PyTorch 2.6 `weights_only=True`.** The DeCAF checkpoint carries OmegaConf
   metadata, which the strict unpickler rejects. Section 5 of the report records
   the same workaround for the vendored Boltz; the fork does not carry it. This
   module forces `weights_only=False` for the duration of the run.

   That permits arbitrary code execution at load time. It is applied here
   because the checkpoint is a published, MIT-licensed research artifact from a
   named group (Genesis Therapeutics, arXiv 2606.08375), downloaded from their
   HuggingFace repo -- not because the flag is harmless.

2. **Import shadowing.** This repository vendors its own `boltz/`. The DeCAF
   script warns that resolving `boltz` to any other install "silently drop[s]
   the trained DeCAF head, and fall[s] back to the teacher diffusion model --
   producing garbage few-step predictions". A silent fallback would look like a
   DeCAF result rather than a bug, so this module puts the fork's `src` first on
   `sys.path` and asserts afterwards that `boltz` actually resolved there.

Usage mirrors `python -m boltz.main`:

    python src/decaf_runner.py predict <inputs> --checkpoint <ckpt> --model boltz1 ...
"""

import functools
import sys
from pathlib import Path

DECAF_SRC = Path.home() / ".boltz" / "decaf" / "repo" / "src"


def _install_fork_on_path():
    if not DECAF_SRC.is_dir():
        raise SystemExit(f"DeCAF fork not found at {DECAF_SRC}; clone the repo first")
    sys.path.insert(0, str(DECAF_SRC))


def _relax_torch_load():
    import torch
    original = torch.load

    @functools.wraps(original)
    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = patched


def main():
    _install_fork_on_path()
    _relax_torch_load()

    import boltz
    resolved = Path(boltz.__file__).resolve()
    if str(DECAF_SRC.resolve()) not in str(resolved):
        raise SystemExit(
            f"`boltz` resolved to {resolved}, not the DeCAF fork at {DECAF_SRC}.\n"
            "Running on would silently use the teacher sampler and produce "
            "few-step output that looks like DeCAF but is not."
        )
    print(f"[decaf_runner] boltz resolved to {resolved}", flush=True)

    from boltz.main import cli
    cli()


if __name__ == "__main__":
    main()
