#!/usr/bin/env bash

source_list=/mnt/sda/coral/beta_status/training_sources.txt
data_root=/media/qimin/samsung1tb/beta_cropped

cat ${source_list} | xargs -n1 -P10 -I {} python ./datasets/geometric_augmentation.py -- --data_root ${data_root} {}