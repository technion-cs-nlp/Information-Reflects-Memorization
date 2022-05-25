import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch import nn
from torchvision import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Dict
from tqdm import tqdm

from train_utils import make_colored_environment, make_shuffled_environment, MLP, mean_nll, mean_accuracy, penalty
import sys
sys.path.append('../../utils')
from information import get_square_mi, get_all_entropies

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_dir, hidden_dim, grayscale_model: bool = False, mode: str = "colored", 
              num_layers:str=2, checkpoint_type=None):
    if checkpoint_type=="last":
        checkpoint = max([int(x.split(".")[0].split("=")[1]) for x in os.listdir(model_dir) if "epoch" in x])
        model_path = model_dir + f"epoch={checkpoint}.pt"
    elif checkpoint_type=="best":
        model_path = model_dir + f"best.pt"
    model = MLP(hidden_dim, grayscale_model, mode, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def get_performance(train_env, val_env, model, dir: str = None, mode: str = "colored"):
    train_logits, val_logits = model(train_env['images']), model(val_env['images'])
    train_loss, val_loss = mean_nll(mode, train_logits, train_env['labels']), mean_nll(mode, val_logits, val_env['labels'])
    train_acc, val_acc = mean_accuracy(mode, train_logits, train_env['labels']), mean_accuracy(mode, val_logits, val_env['labels'])

    if dir:
        with open(dir + "performance.txt", "w") as f:
            f.write(f"Train Loss\tTrain Acc.\tVal Loss\tVal Acc.\n" \
                f"{train_loss}\t{train_acc}\t{val_loss}\t{val_acc}")
    
    return train_loss, val_loss, train_acc, val_acc

def get_reps(args, model: torch.nn.Module, input_img, save_dir: Optional[str] = None) -> Dict:
    """
    Get activations for the given model across the specified dataset.
    Presently, this function is only valid for feed-forward networks from skLearn.
    
    Parameters
    ----------
        model       :       the model as an sklearn.neural_network.MLPClassifier object
        dataloader  :       dataset across which activations need to be computed
    
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
    if args.model_type == "linear":
        if save_dir is not None and os.path.exists(save_dir + "model_reps.dict") and False:
            all_reps = pickle.load(open(save_dir + "model_reps.dict", "rb"))
        else:
            model.eval()
            # num_hidd = len(args.model_conf.split(','))-1
            all_reps = {}
            # for i in range(3):
            #     all_reps[f"fc{i+1}"] = []

            relu = nn.ReLU()
            # for idx, input_img in enumerate(images):
            # all_reps['fc1'].append(model.lin1(input_img.view(input_img.shape[0], 2 * 14 * 14)) \
            #     if grayscale_model else input_img.view(input_img.shape[0], 2, 14 * 14).sum(dim=1))
            # all_reps['fc2'].append(model.lin2(relu(all_reps['fc1'])))
            # all_reps['fc3'].append(model.lin3(relu(all_reps['fc2'])))
            with torch.no_grad():
                all_reps['fc1'] = model.lin1(input_img.view(input_img.shape[0], 2 * 14 * 14)) \
                            if not (args.grayscale_model or args.mode == "shuffled") else model.lin1(input_img.view(input_img.shape[0], 2, 14 * 14).sum(dim=1))
                # print(model.lins)
                for layer, lin in enumerate(model.lins[2::2]):
                    # print(f"Layer: {layer+2}")
                    all_reps[f'fc{layer+2}'] = lin(relu(all_reps[f'fc{layer+1}']))
                all_reps[f'fc{args.num_layers+1}'] = model.linC(relu(all_reps[f'fc{args.num_layers}']))
            
            for key in all_reps.keys():
                # for i, rep in enumerate(all_reps[key]):
                all_reps[key] = all_reps[key].cpu().detach().numpy()
                # all_reps[key] = np.concatenate(all_reps[key], 0)
                print("Reps shape:", all_reps[key].shape)
            
            pickle.dump(all_reps, open(save_dir + "model_reps.dict", "wb"))
        
    elif args.model_type == "resnet":
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
            batches = input_img
            for batch in tqdm(batches):
                model_reps = []
                for layer in range(4):
                    with torch.no_grad():
                        x, y_true = batch
                        x = x.to(device)
                        layer_reps = get_feats(model, x, layer=layer+1)
                        # print(f"Size of representation from layer {layer}: {layer_reps.shape}")
                        model_reps.append(layer_reps.cpu())                 # (bsz, dim)
                model_reps = torch.concat(model_reps, dim=1)                # (bsz, num_l*dim)
                all_reps.append(model_reps)
            all_reps = torch.concat(all_reps, dim=0).numpy()                # (num_samples, num_l*dim)

            np.save(save_dir + "model_reps", all_reps)
    
    return all_reps

def get_entropy(args, reps, save_dir: str, num_bins=20, global_binning=True) -> np.array:
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
    save_file = save_dir + f"entropy_bins={num_bins}"
    if global_binning:
        save_file += "_global"

    if os.path.exists(save_file + ".npy") and False:
        print("Loading pre-computed entropy...")
        entropies = np.load(save_file + ".npy")
    else:
        def compute_entropy(model_type, reps, k, global_binning):
            """
            Organise model representations according to the main function's requirements
            and obtain the N*N MI values
            """
            if model_type == "linear":
                c_reps = np.concatenate([reps[f'fc{i+1}'] for i in range(len(reps)-1)], axis=1)
            else:
                c_reps = reps
            return get_all_entropies(c_reps, num_neighbors=k, to_tqdm=True, global_binning=global_binning)
        
        # compute entropy
        print("Getting entropy...")
        entropies = compute_entropy(args.model_type, reps, num_bins, global_binning)
        
        # save the entropy array as an array
        np.save(save_file, entropies)
        # also print it in a file for easy viewing
        with open(save_file + ".txt", "w") as f:
            print(entropies, file=f)
    
    return entropies

def get_mi(args, reps, save_dir, num_neighbors=3):
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
    save_file = save_dir + "sq_mis"
    
    if os.path.exists(save_file + ".npy"):
        print("Loading pre-computed MI...")
        sq_mis = np.load(save_file + ".npy")
    else:
        def compute_square_mis(reps, num_neighbors):
            """
            Organise model representations according to the main function's requirements
            and obtain the N*N MI values
            """
            c_reps = np.concatenate([reps[f'fc{i+1}'] for i in range(len(reps)-1)], axis=1)
            return get_square_mi(c_reps, num_neighbors, True)

        # compute N*N MI
        print("Getting MI...")
        sq_mis = compute_square_mis(reps, num_neighbors)

        # save the N*N MI
        np.save(save_file, sq_mis)
    
    return sq_mis

def plot_square(mi, save_dir, normalize_mi=False):
    save_file = save_dir + "sq_mis.pdf"

    if os.path.exists(save_file):
        print("Square MI plot exists...")
    else:
        print("Plotting square MIs...")
        if normalize_mi:
            plot = sns.heatmap(mi, vmax=1)
        else:
            plot = sns.heatmap(mi)
        plt.savefig(save_file)

def plot_mean(mi, save_dir, normalize_mi=False, num_layers=3):
    save_file = save_dir + "mean_mis.pdf"
    
    if os.path.exists(save_file):
        print("Mean MI plot exists...")
    else:
        print("Plotting mean MIs...")
        mi = mi.mean(1).reshape(num_layers, -1)
        if normalize_mi:
            plot = sns.heatmap(mi, vmax=1)
        else:
            plot = sns.heatmap(mi)
        plt.savefig(save_file)