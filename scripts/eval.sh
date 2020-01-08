#!/usr/bin/env bash

source_list=../qic003/source_list.txt
epochs=10

cat ${source_list} | xargs -n1 -P10 -I {} python ./obj2sph.py -- --output_folder ${epochs} {}
