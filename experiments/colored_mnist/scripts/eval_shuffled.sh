#!/bin/bash

num_layers=${1:-2}
model="linear";

for ratio in 0.00 0.25 0.50 0.75 1.00; do
    echo "Ratio $ratio"
    python -u main.py \
    --mode="shuffled" \
    --model_type='linear' \
    --hidden_dim=256 \
    --batch_size=512 \
    --l2_regularizer_weight=0.000 \
    --lr=0.001 \
    --penalty_anneal_iters=0 \
    --penalty_weight=0.000 \
    --epochs=500 \
    --early_stopping \
    --ratio=$ratio \
    --save_models \
    --n_restarts=10 \
    --check_performance \
    --num_layers $num_layers \
    --eval_only \
    --eval_model_type="best" \
    --get_entropy # \
    # --get_mi
done;

# bash scripts/eval_shuffled.sh 3 &