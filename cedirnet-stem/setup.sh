#! /usr/bin/env bash
set -euo pipefail

dir=$(realpath "${0%/*}")
model_dir="$TOOLBOX_CACHE/cedirnet-stem"

# CeDiRNet-STEM is currently hosted by ViCoS. CEDIRNET_STEM_REPOSITORY can
# point to an authenticated mirror when the upstream repository is private.
repository=${CEDIRNET_STEM_REPOSITORY:-https://github.com/vicoslab/CeDiRNet-STEM.git}
git clone --depth 1 "$repository" "$model_dir"
cd "$model_dir"
git apply "$dir"/*.patch

install -m 0644 "$dir/annotations.py" "$model_dir/src/datasets/annotations.py"
install -m 0644 "$dir/generic_dataset.py" "$model_dir/src/datasets/GenericPointRadiusDataset.py"
curl --fail --location --retry 3 \
    --output "$model_dir/localization_checkpoint.pth" \
    https://data.vicos.si/skokec/rtfm/CeDiRNet-3DoF/localization_checkpoint.pth

curl --fail --location --retry 3 \
    --output "$model_dir/stem_checkpoint.pt" \
    https://data.vicos.si/skokec/STEM/checkpoint.pth
echo "SHA256 (stem_checkpoint.pt) = b77a30d6346309aeb64a7646d851db74d974758bf7d8e5f2cfcfd9f081637980" | cksum -c

cd "$model_dir"
uv venv --python 3.11
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install \
    opencv-python pandas scikit-learn scikit-image tensorboard matplotlib scipy tqdm \
    segmentation-models-pytorch==0.3.2 future
# SMP 0.3.2 declares timm 0.6.12, which does not import on Python 3.11.
# Install the Python-compatible patch release after SMP, as in the CeDiRNet model.
uv pip install --no-deps timm==0.6.13
uv pip install /opt/apps/modelargs mlflow psutil flask gunicorn /opt/apps/label-studio-ml-backend
