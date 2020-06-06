#!/usr/bin/env bash

source_list=/home/qimin/Downloads/evaluation/evaluate_source.txt
data_root=/mnt/sda/features/gamma/effnetb0_ep12_production
outdir=/home/qimin/Downloads/evaluation/classifier/gamma/effnetb0_ep12_production
epochs=10
weighted=1
loss=log

cat ${source_list} | xargs -n1 -P3 -I {} python ./eval.py \
    -- --outdir ${outdir} \
    --epochs ${epochs} \
    --data_root ${data_root} \
    --weighted ${weighted} \
    --loss ${loss} {}
