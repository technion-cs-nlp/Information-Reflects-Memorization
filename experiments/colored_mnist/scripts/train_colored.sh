#!/bin/bash

model=$1
num_layers=${2:-2}
n_seeds=${3:-10}

for ratio in 0.00 0.125 0.250 0.375 0.500; do
    echo "Ratio $ratio"
    python -u main.py \
    --mode="colored" \
    --model_type=$model \
    --hidden_dim=256 \
    --batch_size=512 \
    --l2_regularizer_weight=0.002 \
    --lr=0.001 \
    --penalty_anneal_iters=0 \
    --penalty_weight=0.000 \
    --epochs=500 \
    --early_stopping \
    --ratio=$ratio \
    --save_models \
    --n_restarts=$n_seeds \
    --check_performance \
    --num_layers $num_layers \
    --get_entropy # \
    # --eval_only # \
    # --get_mi
done;

# bash scripts/train_colored.sh "linear" 3 &