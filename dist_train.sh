#!/usr/bin/env bash
set -x
export PATH=/mnt/petrelfs/share/gcc/gcc-5.4/bin/:/mnt/petrelfs/share/cuda-11.7/bin/:$PATH
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/

CONFIG=$1
GPUS=$2
GPUS_PER_NODE=$3
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

TORCH_DISTRIBUTED_DEBUG=DETAIL \
srun -p INTERN2 \
    ${SRUN_ARGS} \
    --cpus-per-task=12 \
    --gres=gpu:${GPUS_PER_NODE} \
    -n${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --quotatype=spot \
    python -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4}