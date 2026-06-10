#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath ${0%/*})
git clone --depth 1 https://github.com/jerpelhan/GECO2 "$TOOLBOX_CACHE/geco2"
cd "$TOOLBOX_CACHE/geco2"

# git apply "$dir"/*.patch
curl -L -o CNTQG_multitrain_ca44.pt https://huggingface.co/datasets/jerpelhan/geco2-assets/resolve/main/weights/CNTQG_multitrain_ca44.pth?download=true

cp -r Deformable-DETR/models/ops/ models/

uv venv --python 3.10
uv pip install torchvision==0.26.0 torchaudio==2.11.0 numpy==1.26.4 pillow==10.4.0 opencv-python-headless matplotlib scipy scikit-image pycocotools tqdm einops==0.8.1 hydra-core==1.3.2 omegaconf==2.3.0 pandas==2.2.3 imageio==2.37.0
uv pip install gunicorn /opt/apps/modelargs /opt/apps/label-studio-ml-backend