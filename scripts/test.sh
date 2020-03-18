#!/usr/bin/env bash

net_path=/home/qimin/Downloads/evaluation/models/224/resnet101_5eps_best.pt

set -e

python test.py \
    --net efficientnet \
    --net_version b4 \
    --pretrained \
    --dataset coralnet_extraction \
    --input_size 224 \
    --source "s294" \
    --nclasses 1279 \
    --gpu 0 \
    --batch_size 1 \
    --workers 0 \
    --net_path "$net_path" \
    $*
