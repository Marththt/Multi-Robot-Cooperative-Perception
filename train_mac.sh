#!/bin/bash
# Renders the image sets and trains/tests the image-based network on the Mac GPU (MPS).
# Run once from Terminal in the repo folder:  bash train_mac.sh
# Progress: results/bev/train_mac.log (epoch time and validation score per epoch).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
export PYTORCH_ENABLE_MPS_FALLBACK=1
$PY -c "import torch" 2>/dev/null || $PY -m pip install torch
$PY -c "import torch; assert torch.backends.mps.is_available(), 'MPS (Mac GPU) not available'; print('torch', torch.__version__, 'MPS ok')"
mkdir -p results/bev
caffeinate -is bash -c "
  $PY -m bevnet.data --sets train,val,G0,E3,S1 &&
  $PY -m bevnet.train --device mps --workers 6 --train data/bev/train --val data/bev/val \
      --tests 'data/bev/G0_*' 'data/bev/E3_*' 'data/bev/S1_*'
" 2>&1 | tee results/bev/train_mac.log
