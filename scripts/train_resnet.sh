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
    --sets "source" \
    --batch_size 64 \
    --eval_every_train 3 \
    --wdecay 3e-6 \
    --log_time \
    --optim sgd \
    --lr 1e-1 \
    --epoch 25 \
    --workers 24 \
    --logdir "$outdir" \
    $*
