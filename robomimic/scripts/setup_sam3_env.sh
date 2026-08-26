#!/bin/bash
# Creates the `sam3` conda env used by robomimic.models.sam3.Sam3Masker to run
# SAM3 out-of-process (see that module's docstring for why: SAM3 needs
# Python >=3.10 and a recent transformers/huggingface_hub, which conflicts
# with this package's own pins).
#
# Usage: bash robomimic/scripts/setup_sam3_env.sh [env_name]

set -e

ENV_NAME="${1:-sam3}"

CONDA_EXE="${CONDA_EXE:-$(command -v conda)}"
if [ -z "$CONDA_EXE" ]; then
    echo "Error: conda not found. Install miniconda/miniforge first." >&2
    exit 1
fi
CONDA_BASE="$("$CONDA_EXE" info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Conda env '$ENV_NAME' already exists, skipping creation."
else
    conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"
pip install --upgrade pip
pip install numpy pillow torch torchvision "huggingface_hub>=1.5" transformers

python -c "from transformers import Sam3Config, Sam3Model, Sam3Processor; print('SAM3 classes available OK')"

echo ""
echo "sam3 env ready at: $CONDA_BASE/envs/$ENV_NAME/bin/python"
echo "Pass sam3_python=\"$CONDA_BASE/envs/$ENV_NAME/bin/python\" to Sam3Masker if using a non-default env name."
