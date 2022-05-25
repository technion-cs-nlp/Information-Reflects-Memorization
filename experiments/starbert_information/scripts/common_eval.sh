#!/bin/sh
# imported variables
seed=$1
ratio=$2
run=$3
hf_model=${4:-"distilbert-base-uncased"}
data=${5:-"IMDb"}
balance=${6:-"original"}
num_samples=${7:-2500}
probing_layer=${8:-11}
num_train_exs=${9:-10000}
num_test_exs=${10:-5000}

# fixed variables
es_patience=3
es_thresh=0.075
bsz=16
lr=5e-5
layers="0,1,2,3,4,5"
num_neurons=64

nvidia-smi
# run evaluation
python eval_main.py --data_run_type $run --dataset $data \
                    --training_balanced $balance \
                    --ratio $ratio --seed $seed --hf_model $hf_model \
                    --num_samples $num_samples \
                    --train_size $num_train_exs --test_size $num_test_exs \
                    --num_neurons $num_neurons --layers $layers \
                    --get_entropy --global_binning --num_bins 100 \
                    --get_mi --num_neighbors 3 # \
                    # --check_performance --layer_to_probe $probing_layer \
                    # --get_activating_examples \
                    # --plot_square --plot_mean # \
echo "DONE $seed $ratio";