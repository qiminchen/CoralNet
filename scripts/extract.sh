#!/usr/bin/env bash

net_path=/home/qimin/Downloads/resnet50_checkpoint.pt
outdir=/home/qimin/Downloads/evaluation/features/trained_resnet50

set -e

python extract.py \
    --net resnet \
    --net_version resnet50 \
    --pretrained \
    --dataset coralnet_extraction \
    --source "s294" \
    --nclasses 1279 \
    --gpu 0 \
    --batch_size 1 \
    --workers 0 \
    --net_path "$net_path" \
    --logdir "$outdir" \
    $*
