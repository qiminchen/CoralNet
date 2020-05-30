#!/usr/bin/env bash

net_path=/home/qimin/Downloads/evaluation/models/224/effnetb0_10eps_best.pt
outdir=/home/qimin/Downloads/evaluation/features/224/effnetb0_10eps

set -e

python extract.py \
    --net efficientnet \
    --net_version b0 \
    --pretrained \
    --dataset coralnet_extraction \
    --input_size 224 \
    --source "s294" \
    --nclasses 1279 \
    --gpu 0 \
    --batch_size 1 \
    --workers 0 \
    --net_path "$net_path" \
    --logdir "$outdir" \
    $*
