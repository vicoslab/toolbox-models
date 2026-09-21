#!/usr/bin/env bash
set -euo pipefail
dir=$(cd "$(dirname "$0")" && pwd)
: "${TOOLBOX_CACHE:?TOOLBOX_CACHE must be set}"
model_dir="$TOOLBOX_CACHE/cedirnet-stem"
repository=${CEDIRNET_STEM_REPOSITORY:-https://github.com/vicoslab/CeDiRNet-STEM.git}
branch=master
if [[ -e "$model_dir" ]]; then
    printf '%s\n' "Refusing to overwrite existing $model_dir" >&2; exit 1
fi
if [[ -n "${CEDIRNET_STEM_SOURCE:-}" ]]; then
    source_dir=$(realpath "$CEDIRNET_STEM_SOURCE")
    mkdir -p "$model_dir"
    cp -a "$source_dir/src" "$model_dir/src"
    { printf 'local-working-tree=%s\n' "$source_dir"; git -C "$source_dir" branch --show-current; git -C "$source_dir" rev-parse HEAD; git -C "$source_dir" status --short; } > "$model_dir/source-provenance.txt"
else
    # Published stock source; all task/semantic adaptation is plugin-local.
    git clone --depth 1 --branch "$branch" --single-branch "$repository" "$model_dir"
    git -C "$model_dir" rev-parse HEAD > "$model_dir/source-provenance.txt"
fi
# Apply only the existing Python compatibility patch to the disposable copy.
(cd "$model_dir" && git apply "$dir/0001-python-311-collections.patch")
if [[ ${CEDIRNET_STEM_DOWNLOAD_PARTICLES:-1} == 1 ]]; then
    curl --fail --location --retry 3 --output "$model_dir/localization_checkpoint.pth" \
        https://data.vicos.si/skokec/rtfm/CeDiRNet-3DoF/localization_checkpoint.pth
fi
if [[ -n "${CEDIRNET_STEM_ENV:-}" ]]; then
    # Explicit local verified-environment override; does not alter that environment.
    ln -s "$(realpath "$CEDIRNET_STEM_ENV")" "$model_dir/.venv"
else
    uv venv --python 3.11 "$model_dir/.venv"
    python="$model_dir/.venv/bin/python"
    uv pip install --python "$python" torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    # One solve for model + host requirements. Constraints cannot replace the
    # backend's SDK git URL; uv overrides explicitly select our tested SDK.
    # uv splits --override values on spaces even when the shell quotes them.
    # Pass a relative filename from the plugin directory (also for -r).
    (
        cd "$dir"
        uv pip install --python "$python" --override dependency-overrides.txt \
            -r requirements.txt \
            "${CEDIRNET_STEM_MODELARGS:-/opt/apps/modelargs}" \
            "${CEDIRNET_STEM_ML_BACKEND:-/opt/apps/label-studio-ml-backend}"
    )
fi
PYTHONPATH="$model_dir/src:$dir" "$model_dir/.venv/bin/python" -c \
    'import torch, cv2, timm, segmentation_models_pytorch; from label_studio_ml.model import LabelStudioMLBase; from label_studio_sdk.converter.brush import decode_rle; from label_studio_converter.brush import mask2rle; from stem_plugin.stem_tasks import TaskConfig; from stem_plugin.semantic_model import build_semantic_fpn; from stem_plugin.runtime import StemRuntime; print("Verified STEM source:", TaskConfig(False, True).to_dict(), "torch", torch.__version__)'
