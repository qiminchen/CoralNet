#!/usr/bin/env bash

outdir=/mnt/pentagon/qic003/coralnet/result

set -e

python train.py \
    --net efficientnet \
    --net_version "b0" \
    --pretrained \
    --fine_tune \
    --dataset coralnet \
    --nclasses 1279 \
    --gpu -2 \
    --sets "source" \
    --batch_size 256 \
    --eval_every_train 5 \
    --log_time \
    --optim adam \
    --lr 1e-3 \
    --epoch 100 \
    --workers 16 \
    --logdir "$outdir" \
    $*
