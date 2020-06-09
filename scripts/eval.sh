#!/usr/bin/env bash

source_list=/home/qimin/Downloads/evaluation/evaluate_source.txt
data_root=/mnt/sda/features/gamma/effnetb0_4eps
outdir=/home/qimin/Downloads/evaluation/classifier/gamma/effnetb0_4eps_1275_rf
epochs=10
clf_method=rf

cat ${source_list} | xargs -n1 -P1 -I {} python ./eval.py \
    -- --outdir ${outdir} \
    --epochs ${epochs} \
    --data_root ${data_root} \
    --clf_method ${clf_method} {}
