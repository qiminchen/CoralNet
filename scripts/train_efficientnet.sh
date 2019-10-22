#!/usr/bin/env bash

outdir=/mnt/sda/coral/result

set -e

python train.py \
    --net efficientnet \
    --net_version "b0" \
    --pretrained \
    --fine_tune \
    --dataset coralnet \
    --nclasses 1279 \
    --gpu 0 \
    --sets "source" \
    --batch_size 4 \
    --epoch_batches 2500 \
    --eval_batches 1000 \
    --eval_every_train 10 \
    --log_time \
    --optim adam \
    --lr 1e-3 \
    --epoch 1000 \
    --vis_batches_vali 10 \
    --workers 4 \
    --logdir "$outdir" \
    $*
