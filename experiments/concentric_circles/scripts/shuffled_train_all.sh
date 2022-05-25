#!/bin/sh
model=$1
data_size=${2:-100}

division=0.5

# variable variables
for factor in 0.4; do
    for shuffle in 0.00 0.25 0.50 0.75 1.00; do
        bash scripts/common_train.sh $factor $division $shuffle $data_size $model &
    done;
done;

# bash scripts/shuffled_train_all.sh "nn_16_16" &