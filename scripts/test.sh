#!/usr/bin/env bash

net_path=/Users/qiminchen/Downloads/

set -e

python test.py \
    --net efficientnet \
    --net_version "b4" \
    --dataset coralnet \
    --batch_size 32 \
    --workers 4 \
    --net_path "net_path" \
    $*
