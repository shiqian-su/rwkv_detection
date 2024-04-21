#!/usr/bin/env bash

set -x
export PATH=/mnt/petrelfs/share/gcc/gcc-5.4/bin/:/mnt/petrelfs/share/cuda-11.7/bin/:$PATH
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/

CONFIG=$1
GPUS=$2
GPUS_PER_NODE=$3
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p INTERN2 \
    ${SRUN_ARGS} \
    --cpus-per-task=12 \
    --gres=gpu:${GPUS_PER_NODE} \
    -n${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --quotatype=spot \
    --kill-on-bad-exit=1 \
    python -u train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
