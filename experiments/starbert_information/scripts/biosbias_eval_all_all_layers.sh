#!/bin/sh

# fixed variables
hf_model="roberta-base"
layers="0,1,2,3,4,5,6,7,8,9,10,11" # reps only have the last layer
num_neurons=64
num_samples=-1
ratio=0.00

# variable variables
for seed in 0 5 26 42 63; do
    for dataset in raw scrubbed; do
        for balance in original oversampled subsampled; do
            echo "Seed: $seed, Dataset: $dataset, Balance: $balance"
            bash scripts/common_eval.sh $seed $ratio "biosbias" $hf_model BiasInBios-$dataset $balance -1 &
        done;
    done;
done;
echo "DONE ALL"

# bash scripts/biosbias_eval_all_all_layers.sh &