#!/usr/bin/env bash
set -x
export PATH=/mnt/petrelfs/share/gcc/gcc-5.4/bin/:/mnt/petrelfs/share/cuda-11.7/bin/:$PATH
export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/


CONFIG=$1
CHECKPOINT=$2
GPUS=$3
GPUS_PER_NODE=$4
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
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:5}
