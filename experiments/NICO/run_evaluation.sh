#!/bin/bash

set=$1
n_seeds=${2:-5}

nvidia-smi
for seed in `seq 1 $n_seeds`; do
    echo "Seed $seed"
    python -u main.py \
    --training_set=$set \
    --batch_size=64 \
    --epochs=50 \
    --seed=$seed \
    --eval_only \
    --check_performance \
    --get_entropy
done;

# bash run_evaluation.sh "1_plus" 5 &
# bash run_evaluation.sh "2_plus" 5 &