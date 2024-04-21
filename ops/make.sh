#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
set -x
export PATH=/mnt/petrelfs/share/gcc/gcc-5.4/bin/:/mnt/petrelfs/share/cuda-11.7/bin/:$PATH
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/

GPUS=${1}
GPUS_PER_NODE=${2}
SRUN_ARGS=${SRUN_ARGS:-""}

TORCH_DISTRIBUTED_DEBUG=DETAIL \
srun -p INTERN2 \
    ${SRUN_ARGS} \
    --cpus-per-task=12 \
    --gres=gpu:${GPUS_PER_NODE} \
    -n${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --quotatype=spot \
    python setup.py build install
