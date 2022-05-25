#!/bin/bash

set=$1
n_seeds=${2:-5}

for seed in `seq 1 $n_seeds`; do
    python -u main.py \
    --training_set=$set \
    --batch_size=64 \
    --epochs=50 \
    --seed=$seed \
    --check_performance \
    --get_entropy
done;

# bash run_training.sh "1_plus" 5 &
# bash run_training.sh "2_plus" 5 &