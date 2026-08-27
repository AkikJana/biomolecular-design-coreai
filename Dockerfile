# A pinned environment for the folds in this project.
#
# This exists because of a specific failure. The replicate study was run twice --
# once on Apple Silicon CPU with the development fork in ./boltz, once on a
# rented CUDA box where provision.sh ran `pip install boltz` and got the released
# 2.0.3 from PyPI. Those two builds differ in 46 of 106 shared source files,
# including the confidence and diffusion modules. Interface pLDDT's run-to-run SD
# came out 29% apart between them, and because both the hardware and the model
# code changed at once, the cause cannot be recovered. Section 2.4 records it as
# unattributed.
#
# One `pip install boltz` did that. The fork is not on PyPI, its version string
# (2.2.1) is higher than anything published, and nothing in the setup path
# noticed the substitution.
#
# So this image installs ./boltz from the source tree and never from PyPI.
#
# Build (CUDA, matching the rented-box runs):
#   docker build --build-arg TORCH_VARIANT=cu128 -t boltz-fast:cu128 .
# Build (CPU, x86/arm Linux):
#   docker build --build-arg TORCH_VARIANT=cpu -t boltz-fast:cpu .
#
# Run with a GPU:
#   docker run --gpus all -v "$PWD/artifacts:/work/artifacts" boltz-fast:cu128 \
#       python src/seed_variance_study.py --replicates 4
#
# This image does NOT reproduce the Apple Silicon results. Containers cannot
# reach Metal, so a CPU build here is a third environment, not the original one.

FROM python:3.12-slim

ARG TORCH_VARIANT=cu128
ARG TORCH_VERSION=2.8.0

# git: some deps resolve VCS URLs. build-essential: a few wheels compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# torch first and alone. Installing it in the same transaction as boltz lets the
# resolver move it: on the rented box, pulling the cuEquivariance kernels dragged
# torch 2.13/CUDA 13 over 2.8.0+cu128 and broke CUDA outright. Pinning it here,
# before anything can express an opinion about it, is what stops that.
RUN if [ "$TORCH_VARIANT" = "cpu" ]; then \
        pip install --no-cache-dir "torch==${TORCH_VERSION}" \
            --index-url https://download.pytorch.org/whl/cpu ; \
    else \
        pip install --no-cache-dir "torch==${TORCH_VERSION}" \
            --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" ; \
    fi

# The fork, from this tree. Never `pip install boltz`.
COPY boltz /work/boltz
RUN pip install --no-cache-dir --no-deps -e /work/boltz \
 && pip install --no-cache-dir \
        numpy>=1.26 pandas>=2.1 matplotlib>=3.8 scipy scikit-learn \
        biopython gemmi pyyaml pytest==8.3.5

# Fail the build rather than the run if the wrong boltz is importable. The whole
# point of the image is that this cannot silently be upstream's.
RUN python - <<'PY'
import boltz, pathlib
src = pathlib.Path(boltz.__file__).resolve()
assert "/work/boltz/" in str(src), f"boltz resolved to {src}, not the vendored fork"
extra = pathlib.Path("/work/boltz/src/boltz/model/layers/low_rank_pair_representation.py")
assert extra.exists(), "vendored fork is missing its own layers; wrong tree copied"
print(f"  boltz -> {src}")
PY

COPY . /work

# Caches on the image's own filesystem. On RunPod the default put them on
# /workspace, which is MooseFS over FUSE: unpacking Boltz2's 45,228-file
# component store there took 35 minutes against 2.8 seconds on local disk, and
# every fold then read components back across the network.
ENV BOLTZ_CACHE=/work/.cache/boltz \
    TORCH_HOME=/work/.cache/torch \
    HF_HOME=/work/.cache/hf \
    PYTHONPATH=/work:/work/src:/work/boltz/src
# The fused triangle-attention kernels need cuequivariance_ops_torch, which is
# CUDA-only and fails at call time rather than import time. Off by default;
# unset it deliberately if you have installed them and want to measure them.
ENV BOLTZ_NO_KERNELS=1

CMD ["python", "-c", "import torch, boltz; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()}'); print('boltz', boltz.__file__)"]
