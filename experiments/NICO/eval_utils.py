import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch
import os
from typing import Optional
from tqdm import tqdm

import sys
sys.path.append('../../utils')
from information import get_square_mi, get_all_entropies

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_performance(model, dataloader):
    # return model['best_val_metric']
    # return model._metric_attributes
    model.eval()
    model = model.to(device)
    checks = []
    batches = dataloader
    for batch in tqdm(batches):
        x, y_true = batch
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
        # print(y_true[:10], logits[:10])
        checks.extend(list((y_true == torch.argmax(logits, 1)).view(-1)))
    
    accuracy = sum(checks)/len(checks)
    return accuracy
    
def get_reps(model: torch.nn.Module, dataloader, save_dir: Optional[str] = None) -> np.array:
    """
    Get activations for the given model across the specified dataset.
    Presently, this function is only valid for feed-forward networks from skLearn.
    
    Parameters
    ----------
        model       :       the model as an sklearn.neural_network.MLPClassifier object
        dataloader  :       dataset across which activations need to be computed
    
    Returns
    -------
        the model activations as a np array
    """
    if save_dir is not None and os.path.exists(save_dir + "/model_reps.npy"):
        print("Loading pre-computed model activations...")
        all_reps = np.load(save_dir + "/model_reps.npy")
    else:
        print("Computing model activations...")
        def get_feats(model, x, layer=4):
            # See note [TorchScript super()]
            # ResNet class definition: 
            # https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html#resnet50
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)

            if layer == 1:
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                return x
            x = model.layer2(x)
            if layer == 2:
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                return x
            x = model.layer3(x)
            if layer == 3:
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                return x
            
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)

            return x
        
        model.eval()
        model = model.to(device)
        all_reps = []
        batches = dataloader
        for batch in tqdm(batches):
            model_reps = []
            x, y_true = batch
            x = x.to(device)
            # print(f'Sanity check:')
            # print(y_true[:10])
            # print(model(x)[:10])
            # print(torch.argmax(model(x), 1)[:10])
            for layer in range(4):
                with torch.no_grad():
                    layer_reps = get_feats(model, x, layer=layer+1)
                    # print(f"Size of representation from layer {layer}: {layer_reps.shape}")
                    model_reps.append(layer_reps.cpu())                 # (bsz, dim)
            model_reps = torch.concat(model_reps, dim=1)                # (bsz, num_l*dim)
            all_reps.append(model_reps)
        all_reps = torch.concat(all_reps, dim=0).numpy()                # (num_samples, num_l*dim)

        np.save(save_dir + "/model_reps", all_reps)
    
    return all_reps

def get_entropy(reps, save_dir: str, num_bins, global_binning) -> np.array:
    """
    Get neuron-level entropy for the given model across the specified dataset.

    Parameters
    ----------
        reps    :   pre-computed model representations
        save_dir:   the directory in which the entropy values need to be stored
                    these values are stored as a np.array in an .npy file
    
    Returns
    -------
        entropy array that equals the size of network

    """
    save_file = save_dir + f"/entropy_bins={num_bins}"
    if global_binning:
        save_file += "_global"

    if os.path.exists(save_file + ".npy"):
        print("Loading pre-computed entropy...")
        entropies = np.load(save_file + ".npy")
    else:
        def compute_entropy(reps, k, global_binning):
            """
            Organise model representations according to the main function's requirements
            and obtain the N*N MI values
            """
            # c_reps = np.concatenate([reps[:, :] for i in range(reps.shape[0])], axis=1)
            c_reps = reps
            # always select all neurons for all layers,
            # since entropy computation is very quick
            return get_all_entropies(c_reps, num_neighbors=num_bins, 
                                     to_tqdm=True, global_binning=global_binning)
        
        # compute entropy
        print("Getting entropy...")
        entropies = compute_entropy(reps, num_bins, global_binning)

        # save the entropy array as an array
        np.save(save_file, entropies)
        # also print it in a file for easy viewing
        with open(save_file + ".txt", "w") as f:
            print(entropies, file=f)
    
    return entropies

def get_mi(args, reps, save_dir):
    """
    Get neuron-level MI for the given model across the specified dataset.

    Parameters
    ----------
        reps    :   pre-computed model representations
        save_dir:   the directory in which the entropy values need to be stored
                    these values are stored as a np.array in an .npy file
    
    Returns
    -------
        MI array of size N*N, i.e., a square matrix across the 
        total number of neurons in the network

    """
    save_file = save_dir + f"sq_mis_random={args.num_neurons}"
    
    if os.path.exists(save_file + ".npy"):
        print("Loading pre-computed MI...")
        sq_mis = np.load(save_file + ".npy")
    else:
        def compute_square_mis(reps, num_neighbors):
            """
            Organise model representations according to the main function's requirements
            and obtain the N*N MI values
            """
            # c_reps = np.concatenate([reps[:, :args.num_neurons] for x in np.random.randint()], axis=1)
            c_reps = reps[:, np.random.choice(range(reps.shape[1]), 
                                              size=args.num_neurons, 
                                              replace=False)]
            # randomly sampled values from the representation space to compute MI across
            return get_square_mi(c_reps, num_neighbors, True)

        # compute N*N MI
        print("Getting MI...")
        sq_mis = compute_square_mis(reps, args.num_neighbors)

        # save the N*N MI
        np.save(save_file, sq_mis)
    
    return sq_mis

def get_activating_examples(args, reps, entropies, save_dir, top_k_ns=5, top_k_exs=100):
    ...

def get_interventions(args, model, reps, criterion, target, max_num_to_switch, save_dir):
    ...

def plot_square(args, mi, save_dir):
    save_file = f"{save_dir}/sq_mis_layers={args.layers}_numns={args.num_neurons}.pdf"
    print("Plotting square MIs...")
    if args.normalize_mi:
        plot = sns.heatmap(mi, vmax=1)
    else:
        plot = sns.heatmap(mi)
    plt.savefig(save_file)

def plot_mean(args, mi, save_dir):
    save_file = f"{save_dir}/mean_mis_layers={args.layers}_numns={args.num_neurons}.pdf"
    print("Plotting mean MIs...")
    mi = mi.mean(1).reshape(len(args.layers.split(',')), args.num_neurons)
    plot = sns.heatmap(mi)
    plt.savefig(save_file)