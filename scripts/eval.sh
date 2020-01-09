#!/usr/bin/env bash

source_list=../qic003/evaluate_sources.txt
outdir=../qic003/eval/
epochs=10

cat ${source_list} | xargs -n1 -P10 -I {} python ./eval.py -- --outdir ${outdir} --epochs ${epochs} {}
