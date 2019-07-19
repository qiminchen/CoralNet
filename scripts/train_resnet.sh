#!/usr/bin/env bash

outdir=./path/to/output/dir

set -e

python train.py \
    --net resnet50 \
    --dataset coralnet \
    --batch_size 32 \
    --epoch_batches 2500 \
    --eval_batches 5 \
    --log_time \
    --optim adam \
    --lr 1e-3 \
    --epoch 1000 \
    --vis_batches_vali 10 \
    --workers 4 \
    --logdir "$outdir" \
    --suffix '{}' \
    $*
