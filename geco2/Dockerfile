# this dockerfile is just for manually building the wheel for deformable attention
# docker build -q . | xargs docker run --rm --device nvidia.com/gpu=all -v ~/.cache/toolbox:/cache
FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt update && apt install -y python3 python3-pip curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"
ENTRYPOINT ["bash", "-ic", "\
    cd /cache/geco2/Deformable-DETR/models/ops && \
    uv run python setup.py build && \
    uvx patchelf --set-rpath '$ORIGIN/torch/lib' build/lib.linux-x86_64-cpython-310/MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so && \
    uv run python setup.py bdist_wheel \
"]
# result should be whatever /cache/geco2/Deformable-DETR/models/ops/dist/*.whl resolved to
