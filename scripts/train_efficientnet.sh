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
    --batch_size 64 \
    --eval_every_train 2 \
    --wdecay 3e-6 \
    --log_time \
    --optim sgd \
    --lr 1e-1 \
    --epoch 100 \
    --workers 4 \
    --logdir "$outdir" \
    $*
