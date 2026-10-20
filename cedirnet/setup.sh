#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath ${0%/*})

: "${INSTALL_DIR:?Model definition requires newer toolbox version}"
git clone --depth 1 https://github.com/vicoslab/CeDiRNet-3DoF.git "$INSTALL_DIR"
cd "$INSTALL_DIR"
git apply "$dir"/*.patch

uv venv --python 3.12

uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install future opencv_python pandas scikit-learn scikit-image tensorboard matplotlib scipy tqdm segmentation-models-pytorch==0.3.2
uv pip install mlflow psutil /opt/apps/label-studio-ml-backend timm==0.6.13
uv pip install -e /opt/apps/modelargs # use editable install so it works when docker image updates
