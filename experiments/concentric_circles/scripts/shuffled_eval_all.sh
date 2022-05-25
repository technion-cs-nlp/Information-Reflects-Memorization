#!/bin/sh
model=$1

# fixed variables
bsz=16
epochs=1500
ratio=0.75
mean=3
num_seeds=10
division=0.5

nvidia-smi
# variable variables
for seed in 0; do
    for factor in 0.4; do
        for shuffle in 1.0; do
            python eval_main.py --factor $factor --division $division --shuffle $shuffle \
                                --eval_set "in-domain" \
                                --ratio $ratio --mean $mean \
                                --num_seeds $num_seeds --seed $seed \
                                --data_size 100 --model $model \
                                --get_diff_act \
                                --get_entropy --global_binning --num_bins 20 \
                                --get_decision_surface \
                                --get_mi --num_neighbors 3 # \
                                # --plot_square --plot_mean \
                                # --get_interventions --max_num_to_switch 10
        done;
    done;
done;

# bash scripts/shuffled_eval_all.sh "nn_16_16" &