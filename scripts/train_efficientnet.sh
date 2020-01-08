#!/usr/bin/env bash

outdir=../qic003/result

set -e

python train.py \
    --net efficientnet \
    --net_version "b0" \
    --pretrained \
    --fine_tune \
    --dataset coralnet_nautilus \
    --nclasses 1279 \
    --gpu -2 \
    --sets "source" \
    --batch_size 72 \
    --wdecay 3e-6 \
    --log_time \
    --optim sgd \
    --lr 0.05 \
    --max_lr 0.1 \
    --epoch 5 \
    --workers 32 \
    --logdir "$outdir" \
    $*
