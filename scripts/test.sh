#!/usr/bin/env bash

net_path=/mnt/cube/qic003/coral/output/efficientnet/efficientnet_b0_mooreanet_0.001/0/best.pt

set -e

python test.py \
    --net efficientnet \
    --net_version "b0" \
    --dataset coralnet \
    --nclasses 9 \
    --batch_size 32 \
    --workers 4 \
    --net_path "$net_path" \
    $*
