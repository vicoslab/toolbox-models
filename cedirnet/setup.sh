#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath ${0%/*})

git clone --depth 1 https://github.com/vicoslab/CeDiRNet-3DoF.git "$TOOLBOX_CACHE/cedirnet"
cd "$TOOLBOX_CACHE/cedirnet"
git apply "$dir"/*.patch

curl --fail --location --retry 3 \
    --output "$TOOLBOX_CACHE/cedirnet/localization_checkpoint.pth" \
    https://data.vicos.si/skokec/rtfm/CeDiRNet-3DoF/localization_checkpoint.pth
echo "cffcfde184a22c03a67ecc741f3943d0325d4aabe812cb1787796f236403df84  $TOOLBOX_CACHE/cedirnet/localization_checkpoint.pth" | sha256sum --check --status

uv venv --python 3.12

uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install future opencv_python pandas scikit-learn scikit-image tensorboard matplotlib scipy tqdm segmentation-models-pytorch==0.3.2
uv pip install /opt/apps/modelargs mlflow psutil /opt/apps/label-studio-ml-backend timm==0.6.13
