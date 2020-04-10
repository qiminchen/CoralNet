#!/usr/bin/env bash

source_list=/mnt/sda/coral/beta_status/test.txt
data_root=/mnt/sda/beta_cropped

cat ${source_list} | xargs -n1 -P2 -I {} python ./datasets/geometric_augmentation.py -- --data_root ${data_root} {}