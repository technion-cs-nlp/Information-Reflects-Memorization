#!/bin/sh
run=$1
hf_model=${2:-"distilbert-base-uncased"}
data=${3:-"IMDb"}

# fixed variables
epochs=20
es_patience=3
es_thresh=0.075

# variable variables
for seed in `seq 1 5`; do
    echo "Seed $seed";
    for ratio in 0.00 0.20 0.40 0.60 0.80; do
        echo "Ratio $ratio"
        bash scripts/common_eval.sh \
               $seed $ratio $run $hf_model $data &
    done;
done;
echo "DONE ALL";

# bash scripts/common_eval_all.sh "heuristic" "distilbert-base-uncased" &
# bash scripts/common_eval_all.sh "shuffle" "bert-base-uncased" &