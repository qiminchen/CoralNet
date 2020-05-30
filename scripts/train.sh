#!/usr/bin/env bash

outdir=../qic003/result

set -e

python train.py \
    --net resnet \
    --net_version "resnet50" \
    --pretrained \
    --fine_tune \
    --dataset coralnet_nautilus \
    --nclasses 1279 \
    --gpu -2 \
    --batch_size 64 \
    --wdecay 3e-6 \
    --log_time \
    --optim sgd \
    --lr 0.05 \
    --max_lr 0.1 \
    --epoch 5 \
    --workers 24 \
    --logdir "$outdir" \
    $*
