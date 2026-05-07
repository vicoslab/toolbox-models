#! /usr/bin/env bash
set -euo pipefail

git clone --depth 1 https://github.com/facebookresearch/sam3.git "$TOOLBOX_CACHE/sam"
cd "$TOOLBOX_CACHE/sam"
curl https://patch-diff.githubusercontent.com/raw/facebookresearch/sam3/pull/403.patch | git apply -

curl -L https://www.modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt -o sam3.pt
echo "SHA256 (sam3.pt) = 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e" | cksum -c
