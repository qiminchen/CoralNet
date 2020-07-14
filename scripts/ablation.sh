#!/usr/bin/env bash

data_root=/mnt/sda/features/gamma/effnetb0_4eps
outdir=/home/qimin/Downloads/evaluation/classifier/gamma/lr_mlp_ablation
epochs=10
source=s1591

set -e

python ./lr_mlp_ablation.py \
    --source ${source} \
    --outdir ${outdir} \
    --epochs ${epochs} \
    --data_root ${data_root} \
    $*
