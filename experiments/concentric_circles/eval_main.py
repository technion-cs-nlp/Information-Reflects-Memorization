import numpy as np
import pandas as pd
import pickle
import os
import argparse
import sklearn
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import sklearn
from typing import Optional, Dict

from model_utils import evaluate_model, make_train_test_splits
from data_utils import ArtifactFeatures
import sys
sys.path.append('../../utils')
from information import get_square_mi, get_all_entropies
from common_utils import set_seed

from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import expit as logistic_sigmoid
def inplace_relu(X):
    """The ReLu function"""
    np.maximum(X, 0, out=X)
def inplace_logistic(X):
    """The sigmoid function"""
    logistic_sigmoid(X, out=X)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factor', type=float,
                        help = "amount of overlap between the two circles")
    parser.add_argument('--division', type=float,
                        help = "part of the dataset that should follow the heuristic")
    parser.add_argument('--shuffle', type=float,
                        help = "amount of shuffling across the labels")
    parser.add_argument('--ratio', type=float,
                        default=0.75,
                        help = "part of the dataset that shall have spurious features in the input")
    parser.add_argument('--mean', type=int,
                        default=3,
                        help = "the gaussian mean to be used for the heuristic feature")
    parser.add_argument('--model', type=str, 
                        help = "network to be trained; formats: {'linear', 'nn_*'}")
    parser.add_argument('--eval_set', type=str, 
                        choices = ["in-domain", "bias", "anti", "none"],
                        help = "set to be evaluated on")
    parser.add_argument('--num_seeds', default=10, type=int,
                        help = "number of random seeds for each configuration")
    parser.add_argument('--seed', default=42, type=int,
                        help = "random seed for torch, numpy, random")
    parser.add_argument('--num_bins', default=100, type=int,
                        help="number of bins to consider while computing entropy")
    parser.add_argument('--num_neighbors', default=3, type=int,
                        help="number of neigbours to consider while computing MI")
    parser.add_argument('--batch_size', default=64, type=int,
                        help = "batch size")
    parser.add_argument('--learning_rate', default=5e-3, type=float,
                        help = "learning rate")
    parser.add_argument('--data_size', default=20000, type=int,
                        help = "max size of the training data")
    parser.add_argument('--num_samples', default=2000, type=int,
                        help = "max size of the evaluation data now")
    parser.add_argument('--max_num_to_switch', default=10, type=int,
                        help = "maximum number of neurons to switch during interventions")
    
    parser.add_argument('--check_performance', action="store_true")
    parser.add_argument('--get_decision_surface', action="store_true")
    parser.add_argument('--get_entropy', action="store_true")
    parser.add_argument('--global_binning', action="store_true")
    parser.add_argument('--get_mi', action="store_true")
    parser.add_argument('--normalize_mi', action="store_true")
    parser.add_argument('--get_diff_act', action="store_true")
    parser.add_argument('--get_interventions', action="store_true")
    parser.add_argument('--plot_square', action="store_true")
    parser.add_argument('--plot_mean', action="store_true")
    
    return parser

def get_reps(args: argparse.Namespace, model: sklearn.neural_network.MLPClassifier, 
             data: pd.DataFrame, save_dir: Optional[str] = None) -> Dict:
    """
    Get activations for the given model across the specified dataset.
    Presently, this function is only valid for feed-forward networks from skLearn.
    
    Parameters
    ----------
        model       :       the model as an sklearn.neural_network.MLPClassifier object
        data        :       dataset across which activations need to be computed
                            here, it is usually loaded from a .csv file as pd.DataFrame,
                            and consists of the input columns: "x", "y", "c1"
    
    Returns
    -------
        the model activations as a dictionary:
        {
            "input"     :   List[],     # inputs
            "fc_1"      :   List[],     # activations for the 1st layer
                       ...              #                    {i}th  
            "fc_n"      :   List[],     # activations for the nth layer
            "output"    :   List[]      # logits
        }
    """
    if save_dir is not None and os.path.exists(save_dir + f"{args.seed}_model_reps" + ".dict"):
        all_reps = pickle.load(open(save_dir + f"{args.seed}_model_reps.dict", "rb"))
    else:
        all_reps = {'input':[]}
        for i in range(model.n_layers_-2):
            all_reps[f'fc{i+1}'] = []
        all_reps['output'] = []
        for inp in data.values:
            all_reps['input'].append(inp)
            activation = inp
            for i in range(model.n_layers_-1):
                activation = safe_sparse_dot(activation, model.coefs_[i])
                activation += model.intercepts_[i]
                if i != model.n_layers_-2:
                    inplace_relu(activation)
                    all_reps[f'fc{i+1}'].append(activation)
            inplace_logistic(activation)
            all_reps['output'].append(activation)

        for key in all_reps.keys():
            all_reps[key] = np.vstack(all_reps[key])
        
        if save_dir is not None:
            pickle.dump(all_reps, open(save_dir + f"{args.seed}_model_reps.dict", "wb"))
    
    return all_reps

def get_decision_surface(args, data, labels, model, save_dir):
    save_file = f"{save_dir}_{args.seed}_decision_surface"
    
    if not os.path.exists(save_file + '.npy'):
        print('Obtaining decision surface...')
        x, y, z = list(data['x']), list(data['y']), list(data['c1'])

        grid = np.meshgrid(np.arange(min(x), max(x), 0.1),
                        np.arange(min(y), max(y), 0.1),
                        np.arange(min(z), max(z), 0.1), sparse=False)
        xx, yy, zz = grid
        datapoints = {'x':[], 'y':[], 'c1':[]}
        xs, ys, zs = xx.shape[0], yy.shape[1], zz.shape[2]
        for i in range(xs):
            for j in range(ys):
                for k in range(zs):
                    datapoints['x'].append(xx[i,j,k])
                    datapoints['y'].append(yy[i,j,k])
                    datapoints['c1'].append(zz[i,j,k])
        data_ = pd.DataFrame(datapoints)
        model_predicted = model.predict(data_)
        
        preds = model_predicted.reshape(xs, ys, zs)
        data_ = data_.values.reshape(xs, ys, zs, -1)
        coords = [[], [], []]
        for i in range(xs-1):
            for j in range(ys-1):
                for k in range(zs-1):
                    if abs(preds[i,j,k+1]-preds[i,j,k]) == 1 \
                    or abs(preds[i,j+1,k]-preds[i,j,k]) == 1 \
                    or abs(preds[i+1,j,k]-preds[i,j,k]) == 1:
                        c = data_[i,j,k]
                        coords[0].append(c[0])
                        coords[1].append(c[1])
                        coords[2].append(c[2])
        
        coords.append(data.loc[labels==0])
        coords.append(data.loc[labels==1])
        
        np.save(save_file, coords)
    else:
        print('Decision surface exists...')

def get_entropy(args: argparse.Namespace, dataset: pd.DataFrame, 
                model: sklearn.neural_network.MLPClassifier, 
                save_dir: str) -> np.array:
    """
    Get neuron-level entropy for the given model across the specified dataset.

    Parameters
    ----------
        args    :   parsed arguments to extract the parameters
        dataset :   the data across which to compute the entropy
                    here, it is usually loaded from a .csv file as pd.DataFrame,
                    and consists of the input columns: "x", "y", "c1"
        model   :   the model as an sklearn.neural_network.MLPClassifier object
        save_dir:   the directory in which the entropy values need to be stored
                    these values are stored as a np.array in an .npy file
    
    Returns
    -------
        entropy array that equals the size of network

    """
    save_file = f"{save_dir}_{args.seed}_entropy"
    if args.global_binning:
        save_file += "_global"

    if os.path.exists(save_file + ".npy"):
        print("Loading pre-computed entropy...")
        entropies = np.load(save_file + ".npy")
    else:
        # get representations
        print("Getting representations...")
        reps = get_reps(args, model, dataset, save_dir+"_")

        def compute_entropy(reps, k, global_binning):
            """
            Organise model representations according to the main function's requirements
            and obtain the N*N MI values
            """
            c_reps = np.concatenate([reps[f'fc{i+1}'] for i in range(len(reps)-2)], axis=1)
            return get_all_entropies(c_reps, num_neighbors=k, to_tqdm=True, global_binning=global_binning)
        
        # compute entropy
        print("Getting entropy...")
        entropies = compute_entropy(reps, args.num_bins, args.global_binning)

        # save the entropy array as an array
        np.save(save_file, entropies)
        # also print it in a file for easy viewing
        with open(save_file + ".txt", "w") as f:
            print(entropies, file=f)
    
    return entropies

def get_mi(args, dataset, model, save_dir):
    """
    Get neuron-level MI for the given model across the specified dataset.

    Parameters
    ----------
        args    :   parsed arguments to extract the parameters
        dataset :   the data across which to compute the MI
                    here, it is usually loaded from a .csv file as pd.DataFrame,
                    and consists of the input columns: "x", "y", "c1"
        model   :   the model as an sklearn.neural_network.MLPClassifier object
        save_dir:   the directory in which the entropy values need to be stored
                    these values are stored as a np.array in an .npy file
    
    Returns
    -------
        MI array of size N*N, i.e., a square matrix across the 
        total number of neurons in the network

    """
    save_file = f"{save_dir}_{args.seed}_sq_mis"
    
    if os.path.exists(save_file + ".npy"):
        print("Loading pre-computed MI...")
        sq_mis = np.load(save_file + ".npy")
    else:
        # get representations
        print("Getting representations...")
        reps = get_reps(args, model, dataset, save_dir+"_")

        def compute_square_mis(reps, num_neighbors):
            """
            Organise model representations according to the main function's requirements
            and obtain the N*N MI values
            """
            c_reps = np.concatenate([reps[f'fc{i+1}'] for i in range(len(reps)-2)], axis=1)
            c_reps += 1e-2
            return get_square_mi(c_reps, num_neighbors, True)

        # compute N*N MI
        print("Getting MI...")
        sq_mis = compute_square_mis(reps, args.num_neighbors)

        # save the N*N MI
        np.save(save_file, sq_mis)
    
    if args.normalize_mi:
        sq_mis /= np.max(sq_mis)
    return sq_mis

def get_diff_activations(args, factor, model, save_dir):
    save_file = save_dir + f"{args.seed}_diff_activations"

    if os.path.exists(save_file + "_bias-no.npy"):
        print("Loading pre-computed differential activations...")
        bias_no_diff = np.load(save_file + '_bias-no.npy')
        anti_no_diff = np.load(save_file + '_anti-no.npy')
        bias_anti_diff = np.load(save_file + '_bias-anti.npy')
    else:
        print("Computing differential activations...")
        means = (-1*args.mean, args.mean)
        no_bias_data = pd.read_csv(f'data/circles/aligned_none/factor_{factor}_means_{means[0]}_{means[1]}_2000.csv')
        bias_data = pd.read_csv(f'data/circles/aligned_bias/factor_{factor}_means_{means[0]}_{means[1]}_2000.csv')
        anti_bias_data = pd.read_csv(f'data/circles/aligned_anti/factor_{factor}_means_{means[0]}_{means[1]}_2000.csv')
        
        no_bias_reps = get_reps(args, model, no_bias_data[['x', 'y', 'c1']], save_dir+"none_")
        bias_reps = get_reps(args, model, bias_data[['x', 'y', 'c1']], save_dir+"bias_")
        anti_bias_reps = get_reps(args, model, anti_bias_data[['x', 'y', 'c1']], save_dir+"anti_")
        
        def compute_act_diff(reps_1, reps_2, save_file):
            """
            Organise model representations according to the main function's requirements
            and obtain the differntial activations
            """
            reps_1 = np.concatenate([reps_1[f'fc{i+1}'] for i in range(len(reps_1)-2)], axis=1)
            reps_2 = np.concatenate([reps_2[f'fc{i+1}'] for i in range(len(reps_2)-2)], axis=1)
            reps_diff = np.subtract(reps_1, reps_2)
            np.save(save_file, reps_diff)

            return reps_diff
        
        bias_no_diff = compute_act_diff(bias_reps, no_bias_reps, save_file + '_bias-no')
        anti_no_diff = compute_act_diff(anti_bias_reps, no_bias_reps, save_file + '_anti-no')
        bias_anti_diff = compute_act_diff(bias_reps, anti_bias_reps, save_file + '_bias-anti')

    return bias_no_diff, anti_no_diff, bias_anti_diff

def get_interventions(args, factor, model, criterion, target, max_num_to_switch, save_dir):
    save_file = save_dir + f"{args.seed}_intervene_switch"
    
    print(f"Performing intervention on {criterion}...")
    print("\t", end="")
    if criterion=="diff_acts":
        query_reps = get_diff_activations(args, factor, model, save_dir)[0]
        query_reps = np.mean(query_reps, 0)
    elif criterion=="entropy":
        query_reps = get_entropy(args, None, model, save_dir + "bias")
    elif criterion=="mi":
        query_reps = get_mi(args, None, model, save_dir + "bias")
        query_reps = np.mean(query_reps, 1)
    
    no_bias_data = pd.read_csv(f'data/circles/aligned_none/factor_{factor}_means_{means[0]}_{means[1]}_{args.data_size}.csv')
    bias_data = pd.read_csv(f'data/circles/aligned_bias/factor_{factor}_means_{means[0]}_{means[1]}_{args.data_size}.csv')
    anti_bias_data = pd.read_csv(f'data/circles/aligned_anti/factor_{factor}_means_{means[0]}_{means[1]}_{args.data_size}.csv')
    
    m_no_bias, m_bias, m_anti_bias \
    = model.score(no_bias_data[['x', 'y', 'c1']], no_bias_data[['label']]), \
    model.score(bias_data[['x', 'y', 'c1']], bias_data[['label']]), \
    model.score(anti_bias_data[['x', 'y', 'c1']], anti_bias_data[['label']])

    def switch_off_neurons(model, neuron_posns, model_name):
        neurons_per_layer = [int(x) for x in model_name.split('_')[1:]]

        model_ = copy.deepcopy(model)
        weights, biases = model_.coefs_, model_.intercepts_
        for pos in neuron_posns:
            neuron_posn = pos
            for i, num_ns in enumerate(neurons_per_layer):
                if neuron_posn - num_ns < 0:
                    layer_posn = i
                    break
                neuron_posn -= num_ns
            
            for i, _ in enumerate(weights[layer_posn]):
                weights[layer_posn][i][neuron_posn] = 0.
            biases[layer_posn][neuron_posn] = -0.1
            
            for i, _ in enumerate(weights[layer_posn+1][neuron_posn]):
                weights[layer_posn+1][neuron_posn][i] = 0.
        
        model_.coefs_, model_.intercepts_ = weights, biases
        return model_
    
    for num_to_switch in range(1, max_num_to_switch):
        if target=="lower":
            to_switch_off = query_reps.argsort()[-num_to_switch:]
        elif target=="higher":
            to_switch_off = query_reps.argsort()[:num_to_switch]
        elif target=="random":
            to_switch_off = np.random.choice(query_reps.argsort()[3:-3], num_to_switch)
        
        model_off = switch_off_neurons(model, to_switch_off, args.model)
        moff_no_bias, moff_bias, moff_anti_bias \
        = model_off.score(no_bias_data[['x', 'y', 'c1']], no_bias_data[['label']]), \
        model_off.score(bias_data[['x', 'y', 'c1']], bias_data[['label']]), \
        model_off.score(anti_bias_data[['x', 'y', 'c1']], anti_bias_data[['label']])
        
        with open(save_file + ".txt", "a+") as f:
            # if len(f.readlines()) < 1:
            #     f.write("criterion\ttarget\tno-bias(o)\tno-bias(i)\tbias(o)\tbias(i)\tanti-bias(o)\tanti-bias(i)\n")
            #     f.write("---------\t------\t----------\t----------\t-------\t-------\t------------\t------------\n")
            f.write(f"{criterion}\t{target}\t{num_to_switch}\t{m_no_bias}\t{moff_no_bias}\t{m_bias}\t{moff_bias}\t{m_anti_bias}\t{moff_anti_bias}\n")

def plot_square(args, mi, save_dir):
    save_file = f"{save_dir}_{args.seed}_sq_mis.pdf"

    if os.path.exists(save_file):
        print("Square MI plot exists...")
    else:
        print("Plotting square MIs...")
        if args.normalize_mi:
            plot = sns.heatmap(mi, vmax=1)
        else:
            plot = sns.heatmap(mi)
        plt.savefig(save_file)

def plot_mean(args, mi, save_dir):
    save_file = f"{save_dir}_{args.seed}_mean_mis.pdf"
    
    if os.path.exists(save_file):
        print("Mean MI plot exists...")
    else:
        print("Plotting mean MIs...")
        mi = mi.mean(1).reshape(len(args.model.split('_'))-1, -1)
        if args.normalize_mi:
            plot = sns.heatmap(mi, vmax=1)
        else:
            plot = sns.heatmap(mi)
        plt.savefig(save_file)

if __name__ == '__main__':
    # Cross-check GPU avilability
    print(f"GPU(s) available? {torch.cuda.is_available()}")

    # parse args
    args = init_parser().parse_args()
    # divisions = [float(x) for x in args.divisions.split(',')]
    # factors = [float(x) for x in args.factors.split(',')]
    shuffle, division, factor = args.shuffle, args.division, args.factor
    means = (-args.mean, args.mean)

    print(f"evaluation for {args.model} {division} artifacts model on {args.eval_set}:")

    # get dataset
    data_object = ArtifactFeatures(args.ratio, factor, division, 0.00,
                                  (-1*args.mean, args.mean), args.num_samples)
    if args.eval_set in ["bias", "anti", "none"]:
        index = ["bias", "anti", "none"].index(args.eval_set)
        factor_data = data_object.get_dataset(aligned=True, seed=21)[index]
    elif args.eval_set == "in-domain":
        factor_data = data_object.get_dataset(artifact="bias", aligned=False, seed=21, evaluation=True)
    
    # set_seed(args.seed)
    # _, _, data, labels = make_train_test_splits(factor_data, ['x', 'y', 'c1'])
    data, labels = factor_data[['x', 'y', 'c1']], factor_data['label']

    # get model
    save_dir = f'models/factor={factor}_division={division}_shuffle={shuffle}_size={args.data_size}'
    save_name = f'{save_dir}/{args.model}'
    print(f"{save_name}/{args.seed}.pkl")
    assert os.path.exists(f"{save_name}/{args.seed}.pkl")

    model = pickle.load(open(f'{save_name}/{args.seed}.pkl', 'rb'))
    
    # evaluate model performance
    if args.check_performance:
        acc = f"{evaluate_model(save_name, data, num_seeds=args.num_seeds, feats=['x', 'y', 'c1'])}"
        print(f"\tsep: {factor} {args.model} {acc}")
    
    #  get decision surface
    if args.get_decision_surface:
        get_decision_surface(args, data, labels, model, f"{save_name}/{args.eval_set}")
        # for un-aligned biased data---same as the training distribution
        data_file = f'{save_dir}/dataset.csv'
        factor_data = pd.read_csv(data_file)
        set_seed(args.seed)
        _, _, unaligned_data, unaligned_labels = make_train_test_splits(factor_data, ['x', 'y', 'c1'])
        get_decision_surface(args, unaligned_data, unaligned_labels, model, f"{save_name}/unaligned_bias")

    # get entropy
    if args.get_entropy:
        h = get_entropy(args, data, model, f"{save_name}/{args.eval_set}")
    
    # get MI
    if args.get_mi:
        mi = get_mi(args, data, model, f"{save_name}/{args.eval_set}")
    
    # plot square plot (to check cluster)
    if args.plot_square:
        plot_square(args, mi, f"{save_name}/{args.eval_set}")
        plt.close()
    
    # plot mean plot
    if args.plot_mean:
        plot_mean(args, mi, f"{save_name}/{args.eval_set}")
        
    # get differential activations
    if args.get_diff_act:
        _, _, _ = get_diff_activations(args, factor, model, f"{save_name}/")
    
    # get model performance on intervention
    if args.get_interventions:
        get_interventions(args, factor, model, "diff_acts", "higher", args.max_num_to_switch, f"{save_name}/")
        get_interventions(args, factor, model, "diff_acts", "random", args.max_num_to_switch, f"{save_name}/")
        get_interventions(args, factor, model, "entropy", "lower", args.max_num_to_switch, f"{save_name}/")
        get_interventions(args, factor, model, "entropy", "random", args.max_num_to_switch, f"{save_name}/")
        get_interventions(args, factor, model, "mi", "higher", args.max_num_to_switch, f"{save_name}/")
        get_interventions(args, factor, model, "mi", "random", args.max_num_to_switch, f"{save_name}/")