#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath ${0%/*})
: "${INSTALL_DIR:?Model definition requires newer toolbox version}"
git clone --depth 1 https://github.com/jerpelhan/GECO2 "$INSTALL_DIR"
cd "$INSTALL_DIR"

cp -r Deformable-DETR/models/ops/ models/

uv venv --python 3.11
uv pip install torchvision==0.26.0 torchaudio==2.11.0 numpy==1.26.4 pillow==10.4.0 opencv-python-headless matplotlib scipy scikit-image pycocotools tqdm einops==0.8.1 hydra-core==1.3.2 omegaconf==2.3.0 pandas==2.2.3 imageio==2.37.0
uv pip install gunicorn /opt/apps/label-studio-ml-backend https://github.com/vicoslab/toolbox-models/releases/download/slaif-1.1.1/multiscaledeformableattention-1.0-cp311-cp311-linux_x86_64.whl
uv pip install -e /opt/apps/modelargs # use editable install so it works when docker image updates