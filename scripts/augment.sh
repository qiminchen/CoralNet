#!/usr/bin/env bash

source_list=/home/qimin/Downloads/evaluation/evaluate_sources.txt
data_root=/home/qimin/Downloads/evaluation/images_augmented

cat ${source_list} | xargs -n1 -P3 -I {} python ./datasets/geometric_augmentation.py -- --data_root ${data_root} {}