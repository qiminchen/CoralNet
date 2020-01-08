#!/usr/bin/env bash

source_list=../qic003/evaluate_sources.txt
epochs=10

cat ${source_list} | xargs -n1 -P10 -I {} python ./obj2sph.py -- --epochs ${epochs} {}
