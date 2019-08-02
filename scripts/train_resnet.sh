#!/usr/bin/env bash

outdir=/mnt/cube/qic003/coral/output/resnet

set -e

python train.py \
    --net resnet \
    --net_version "resnet50" \
    --pretrained \
    --fine_tune \
    --dataset coralnet \
    --gpu 0 \
    --sets "source" \
    --batch_size 32 \
    --epoch_batches 2500 \
    --eval_batches 5 \
    --log_time \
    --optim adam \
    --lr 1e-3 \
    --epoch 30 \
    --vis_batches_vali 10 \
    --workers 4 \
    --logdir "$outdir" \
    $*
