#!/bin/bash
# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

. ../../path.sh

if [ $# -lt 1 ]; then
    echo 'One argument required (librimix_dir)'
    exit 1;
fi

librimix_dir=$1

# Overall metadata (by Asteroid recipes)
python local/create_local_metadata.py --librimix_dir $librimix_dir

# Enrollment utterances for test and dev
python local/create_enrollment_csv_fixed.py \
    data/wav8k/min/test/mixture_test_mix_both.csv \
    data/wav8k/min/test/map_mixture2enrollment \
    data/wav8k/min/test/mixture2enrollment.csv
python local/create_enrollment_csv_fixed.py \
    data/wav8k/min/dev/mixture_dev_mix_both.csv \
    data/wav8k/min/dev/map_mixture2enrollment \
    data/wav8k/min/dev/mixture2enrollment.csv

# Enrollment utterances for training
python local/create_enrollment_csv_all.py \
    data/wav8k/min/train-100/mixture_train-100_mix_both.csv \
    data/wav8k/min/train-100/mixture2enrollment.csv
python local/create_enrollment_csv_all.py \
    data/wav8k/min/train-360/mixture_train-360_mix_both.csv \
    data/wav8k/min/train-360/mixture2enrollment.csv
