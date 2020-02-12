#!/usr/bin/env bash

objdir=/mnt/sda/coral/status/training_image_all.txt

cat ${objdir} | xargs -n1 -P10 -I {} python ./datasets/cropimage.py {}