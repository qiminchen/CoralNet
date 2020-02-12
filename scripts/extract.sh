#!/usr/bin/env bash

net_path=/home/qimin/Downloads/resnet50.pt
outdir=/media/qimin/seagate5tb/evaluation/features

set -e

python extract.py \
    --net resnet \
    --net_version "resnet50" \
    --pretrained \
    --dataset coralnet_extraction \
    --source "s16" \
    --nclasses 1279 \
    --gpu 0 \
    --batch_size 1 \
    --workers 4 \
    --net_path "$net_path" \
    --logdir "$outdir"
    $*
