#!/bin/bash
# Build the project virtualenv with uv + SAM 2 + Hiera-L checkpoint, entirely under /scratch.
# Run on the LOGIN node (needs internet). Usage: bash scripts/setup_env.sh
set -euo pipefail
ROOT=/scratch/b23cs1017/cv
VENV=$ROOT/.venv
export UV_INSTALL_DIR=$ROOT/.tools/bin           # uv binary
export UV_CACHE_DIR=$ROOT/.cache/uv              # package cache
export UV_PYTHON_INSTALL_DIR=$ROOT/.tools/python # uv-managed CPython
export UV_LINK_MODE=copy TMPDIR=$ROOT/.cache/tmp
mkdir -p $UV_CACHE_DIR $TMPDIR $ROOT/checkpoints
export PATH=$UV_INSTALL_DIR:$PATH

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 sh
[ -x $VENV/bin/python ] || uv venv --python 3.11 $VENV
export VIRTUAL_ENV=$VENV

# cu128 wheels cover Blackwell (sm_120, cn07 RTX PRO 6000) as well as older GPUs.
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install scipy opencv-python-headless tifffile imagecodecs pillow tqdm pyyaml tensorboard pytest setuptools wheel
# SAM 2 (skip the optional CUDA extension; only the image encoder is used).
SAM2_BUILD_CUDA=0 uv pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"

CK=$ROOT/checkpoints/sam2.1_hiera_large.pt
[ -s $CK ] || curl -L -o $CK https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
$VENV/bin/python -c "import torch, sam2, scipy; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'sam2 ok')"
ls -lh $CK
