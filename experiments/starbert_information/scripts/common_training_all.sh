#!/bin/sh
run=$1
hf_model=${2:-"distilbert-base-uncased"}
lr=${3:-5e-4}
bsz=${4:-32}

# fixed variables
dataset="IMDb"
es_patience=3
es_thresh=0.075

# variable variables
for seed in `seq 1 5`; do
    echo "Seed $seed";
    for ratio in 0.00 0.20 0.40 0.60 0.80; do
        echo "Ratio $ratio"
        bash scripts/common_training.sh \
             $seed $ratio $run $hf_model $lr $bsz &
    done;
done;
echo "DONE ALL";

# bash scripts/common_training_all.sh "heuristic" "distilbert-base-uncased" 5e-5 16 &
# bash scripts/common_training_all.sh "shuffle" "distilbert-base-uncased" 5e-5 16 &