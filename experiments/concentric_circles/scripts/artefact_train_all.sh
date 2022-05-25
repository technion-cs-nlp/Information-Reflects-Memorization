#!/bin/sh
model=$1
data_size=${2:-2000}

shuffle=0.0

# variable variables
for factor in 0.4; do
    for division in 0.5 0.65 0.75 0.9 1.0; do
        bash scripts/common_train.sh $factor $division $shuffle $data_size $model &
    done;
done;

# bash scripts/artefact_train_all.sh "nn_16_16" &