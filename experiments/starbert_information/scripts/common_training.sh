#!/bin/sh
# imported variables
seed=$1
ratio=$2
run=$3
hf_model=${4:-"distilbert-base-uncased"}
lr=${5:-5e-4}
bsz=${6:-32}
dataset=${7:-"IMDb"}
train_size=${8:-50000}
test_size=${9:-5000}

# fixed variables
epochs=200
es_patience=3
es_thresh=0.075
num_neurons=64
layers="0,1,2,3,4,5"

nvidia-smi
# run training
export WANDB_PROJECT=$hf_model-$dataset-$run
python trainer_main.py --data_run_type $run --dataset $dataset \
                    --ratio $ratio --seed $seed --hf_model $hf_model \
                    --batch_size $bsz --learning_rate $lr \
                    --max_epochs $epochs --stopping_patience $es_patience \
                    --stopping_train_thresh $es_thresh \
                    --train_size $train_size --test_size $test_size # \
                    # --load_last_checkpoint

echo "DONE $seed $ratio";