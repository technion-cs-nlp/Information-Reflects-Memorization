factor=$1
division=$2
shuffle=$3
data_size=$4
model=$5

# fixed variables
bsz=16
epochs=10000
ratio=0.75
mean=3
num_seeds=5

nvidia-smi

python trainer_main.py --factor $factor --division $division --shuffle $shuffle \
                       --ratio $ratio --mean $mean \
                       --max_epochs $epochs --num_seeds $num_seeds \
                       --data_size $data_size --model $model

seed=0

python eval_main.py --factor $factor --division $division --shuffle $shuffle \
                    --eval_set "in-domain" \
                    --ratio $ratio --mean $mean \
                    --num_seeds $num_seeds --seed $seed \
                    --data_size $data_size --model $model \
                    --get_diff_act \
                    --get_entropy --global_binning --num_bins 20 \
                    --get_decision_surface \
                    --get_mi --num_neighbors 3 # \
                    # --plot_square --plot_mean \
                    # --get_interventions --max_num_to_switch 10