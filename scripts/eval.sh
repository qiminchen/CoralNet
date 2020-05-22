#!/usr/bin/env bash

source_list=/home/qimin/Downloads/evaluation/evaluate_sources.txt
data_root=/home/qimin/Downloads/evaluation/features/gamma/effnetb0_3eps_geometric
outdir=/home/qimin/Downloads/evaluation/classifier/gamma/effnetb0_3eps_geometric_svm
epochs=10
loss=hinge

cat ${source_list} | xargs -n1 -P1 -I {} python ./eval.py -- --outdir ${outdir} --epochs ${epochs} --data_root ${data_root} --loss ${loss} {}
