#!/usr/bin/env bash

net_path=/Users/qiminchen/Downloads/best.pt
outdir=/Users/qiminchen/Downloads

set -e

python extract.py \
    --net efficientnet \
    --net_version "b0" \
    --pretrained \
    --dataset coralnet \
    --source "s102" \
    --nclasses 9 \
    --sets "source" \
    --gpu -1 \
    --batch_size 32 \
    --workers 4 \
    --net_path "$net_path" \
    --logdir "$outdir"
    $*
