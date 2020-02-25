#!/usr/bin/env bash

source_list=/home/qimin/Downloads/evaluation/evaluate_sources_294.txt
data_root=/home/qimin/Downloads/evaluation/features/trained_resnet50
outdir=/home/qimin/Downloads/evaluation/classifier/trained_resnet50
epochs=10

cat ${source_list} | xargs -n1 -P3 -I {} python ./eval_local.py -- --outdir ${outdir} --epochs ${epochs} --data_root ${data_root} {}
