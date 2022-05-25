import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch
from torch import nn
import os
from typing import Optional
import pickle
import json
import copy

import datasets
from datasets import load_metric
from transformers import RobertaModel, AutoModelForSequenceClassification, TrainingArguments, Trainer

import get_shuffle_data
import get_heuristic_data
import get_feather_data
import get_biasinbios_data

import sys
sys.path.append('../../utils')
from information import get_square_mi, get_all_entropies
from probing import run_MDL_probing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(eval_preds):
    """
    Obtain metric values for evaluation
    """
    metric = load_metric("accuracy")
    logits, labels = eval_preds.predictions, eval_preds.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_run_type', type=str, 
                        choices=["shuffle", "shuffle_", "shuffle_larger", "heuristic", "hans", "biosbias"],
                        help = "Type of dataset to perform training")
    parser.add_argument('--check_performance', action="store_true")
    parser.add_argument('--get_entropy', action="store_true")
    parser.add_argument('--global_binning', action="store_true")
    parser.add_argument('--get_mi', action="store_true")
    parser.add_argument('--get_activating_examples', action="store_true")
    parser.add_argument('--get_interventions', action="store_true")
    parser.add_argument('--normalize_mi', action="store_true")
    parser.add_argument('--plot_square', action="store_true")
    parser.add_argument('--plot_mean', action="store_true")
    parser.add_argument('--compute_again', action="store_false", 
                        help="whether to overwrite existing file (specify to do so)")
    
    parser.add_argument('--top_k_ns', default=5, type=int,
                        help="value of top-k for which to extract activating exs")
    parser.add_argument('--max_num_to_switch', default=10, type=int,
                        help = "maximum number of neurons to switch during interventions")

    parser.add_argument('--dataset', type=str, 
                        help = "Type of dataset to perform training",
                        choices=['IMDb', 'Amazon', 'CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE', 'HANS', \
                        "BiasInBios-raw", "BiasInBios-scrubbed"])
    parser.add_argument('--training_balanced', type=str, 
                        help = "[Only for BiasInBios] Type of de-biasing used",
                        choices=['na', 'original', 'oversampled', 'subsampled'], default='na')
    parser.add_argument('--ratio', type=float,
                        default=0.00,
                        help = "heuristic or shuffle ratio")
    parser.add_argument('--hf_model', type=str, 
                        help = "HugginFace model to be used for training and tokenizing",
                        choices=["distilbert-base-uncased", "bert-base-uncased", "roberta-base"])
    parser.add_argument('--seed', default=42, type=int,
                        help = "random seed for torch, numpy, random")
    parser.add_argument('--num_neighbors', default=10, type=int,
                        help="number of neigbours to consider while computing MI or entropy")
    parser.add_argument('--num_bins', default=10, type=int,
                        help="number of bins in which to discretize while computing MI or entropy")
    
    parser.add_argument('--max_epochs', default=10, type=int,
                        help = "max training epochs without early stopping")
    parser.add_argument('--stopping_patience', default=3, type=int,
                        help = "patience before early stopping")
    parser.add_argument('--stopping_train_thresh', default=0.08, type=float,
                        help = "value training loss must reach before early stopping is checked")
    parser.add_argument('--batch_size', default=64, type=int,
                        help = "batch size")
    parser.add_argument('--learning_rate', default=5e-3, type=float,
                        help = "learning rate")
    
    parser.add_argument('--num_samples', default=2500, type=int,
                        help = "number of samples across which to compute the MI (-1 -> entire set)")
    parser.add_argument('--num_neurons', default=64, type=int,
                        help = "number of neurons per layer for which to compute the MI")
    parser.add_argument('--layers', default='0,1,2,3,4,5', type=str,
                        help = "layers for whom to compute the MI")
    # parser.add_argument('--checkpoint', default=2000, type=int,
    #                     help = "model checkpoint to load")
    parser.add_argument('--train_size', default=50000, type=int,
                        help = "max size of training data")
    parser.add_argument('--test_size', default=10000, type=int,
                        help = "max size of testing data")
    parser.add_argument('--layer_to_probe', type=int, default=11,
                        help = "Layer to probe for bias in bios")
    
    return parser

def get_dataset(args, type_=None, dir=None, suffix_custom=None, layer=None):
    if "shuffle" in args.data_run_type:
        train_dataset, val_dataset = get_shuffle_data.get_data(args.seed, args.dataset, args.ratio, 
                                                               args.hf_model, args.train_size, args.test_size)
    elif args.data_run_type == "heuristic":
        train_dataset, val_dataset, test_dataset = get_heuristic_data.get_data(args.seed, args.ratio, args.hf_model,
                                                                               args.train_size, args.test_size)
    elif args.data_run_type == "hans":
        labels_dict = {"contradiction": 0, "entailment": 1, "neutral": 2} if args.dataset == "MNLI" else None
        train_dataset = get_feather_data.get_data(args.seed, args.dataset, args.hf_model, 
                                                  "train", labels_dict)
        val_dataset = get_feather_data.get_data(args.seed, args.dataset, args.hf_model, 
                                                "test", labels_dict)
        test_dataset = val_dataset
    elif args.data_run_type == "biosbias":
        num_layers = len(args.layers.split(','))
        datasets = []
        for split in ["train", "valid", "test"]:
            if dir is None:
                data_path = f"../../../GenderBias/data/biosbias/tokens_raw_original_roberta-base_128_{split}-seed_{args.seed}.pt"
            else:
                if suffix_custom is not None:
                    data_path = dir + suffix_custom + f"{split}.pt"
                else:
                    if num_layers == 1:
                        data_path = dir + f'/vectors_raw_original_roberta-base_128_{split}-seed_{args.seed}.pt'
                    else:
                        data_path = dir + f"/vectors_raw_original_roberta-base_all_layers_128_{split}-seed_{args.seed}.pt"
            datasets.append(get_biasinbios_data.get_data(data_path, type_, None, layer=layer))
        train_dataset, val_dataset, test_dataset = tuple(datasets)
        if dir is not None:
            return train_dataset, val_dataset, test_dataset
    
    return train_dataset, val_dataset

def get_model(args):
    # Load model and datasets
    if args.data_run_type == "hans":
        dir = f"models/{args.data_run_type}/{args.hf_model}/bert_0{args.seed}" if args.seed < 10 \
            else f"models/{args.data_run_type}/{args.hf_model}/bert_{args.seed}"
        model = AutoModelForSequenceClassification.from_pretrained(
            f"{dir}").to(device)
    elif "shuffle" in args.data_run_type or args.data_run_type == "heuristic":
        if args.dataset != "IMDb":
            dir = f"models/{args.data_run_type}/{args.hf_model}/seed={args.seed}/dataset={args.dataset}_size={args.train_size}_ratio={args.ratio:.2f}"
        else:
            dir = f"models/{args.data_run_type}/{args.hf_model}/seed={args.seed}/ratio_{args.ratio:.2f}"
        checkpoint = -1
        for f in os.listdir(dir):
            if "checkpoint" in f:
                checkpoint = max(checkpoint, int(f.split("-")[1]))
        print("Checkpoint:", checkpoint)

        model = AutoModelForSequenceClassification.from_pretrained(
            f"{dir}/checkpoint-{checkpoint}").to(device)
    elif args.data_run_type == "biosbias":
        class roBERTa_classifier(nn.Module):
            def __init__(self, n_labels):
                super().__init__()
                self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False, output_attentions=False,
                                                            output_hidden_states=False)
                self.classifier = nn.Linear(768, n_labels)

            def forward(self, x, attention_mask, return_features=False):
                features = self.roberta(x, attention_mask=attention_mask)[0]
                x = features[:, 0, :]
                x = self.classifier(x)
                if return_features:
                    return x, features[:, 0, :]
                else:
                    return x
        model_path = f"../../../GenderBias/src/checkpoints/bias_in_bios/roberta-base/finetuning/{args.dataset.split('-')[1]}/{args.training_balanced}/seed_{args.seed}/best_model.pt"
        model = roBERTa_classifier(28)
        state_dict = torch.load(model_path, map_location=device)
        if callable(state_dict['model_state_dict']):
            model.load_state_dict(state_dict['model_state_dict']())
        else:
            model.load_state_dict(state_dict['model_state_dict'])
    
    return model

def check_performance(args, val_dataset, model, train_dataset=None, test_dataset=None, save_dir=None):
    if args.data_run_type == "biosbias":
        compression = run_MDL_probing(args.seed, (train_dataset, val_dataset, test_dataset), 768, 
                                        "mlp", args.batch_size, shuffle=False, save_dir=save_dir)
        return compression
    
    # check performance if asked
    batch_size = args.batch_size

    training_args = TrainingArguments(
        output_dir=save_dir,                    # output directory
        evaluation_strategy='steps',            # evaluate by steps, not epochs
        eval_steps=100,                         # evaluate after these many steps
        save_total_limit=3,                     # only save these many model checkpoints
        num_train_epochs=args.max_epochs,       # total number of training epochs
        per_device_train_batch_size=batch_size, # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        warmup_steps=100,                       # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                      # strength of weight decay
        learning_rate=args.learning_rate,       # learning rate
        logging_dir=save_dir,                   # directory for storing logs
        logging_steps=10,
        push_to_hub=False,
        metric_for_best_model='loss',           # metric for early stopping comp.
        load_best_model_at_end=True             # required for early stopping
    )

    trainer = Trainer(
        model=model,                            # the instantiated 🤗 Transformers model to be trained
        args=training_args,                     # training arguments, defined above
    )

    val_predictions = trainer.predict(val_dataset)
    val_performance = compute_metrics(val_predictions)
    print(f"Validation Performance for Seed {args.seed} and Ratio {args.ratio}: {val_performance}")
    with open(f"{save_dir}/val_performances.txt", "a+") as f:
        f.write(f"{args.ratio:.2f}:\t{val_performance['accuracy']}\n")

    if train_dataset is not None:
        train_predictions = trainer.predict(train_dataset)
        train_performance = compute_metrics(train_predictions)
        print(f"Training Performance for Seed {args.seed} and Ratio {args.ratio}: {train_performance}")
        with open(f"{save_dir}/train_performances.txt", "a+") as f:
            f.write(f"{args.ratio:.2f}:\t{train_performance['accuracy']}\n")
        
        return train_performance, val_performance
    
    return val_performance
    
def get_reps(args, model: Optional[torch.nn.Module]=None, 
             dataset: Optional[datasets.Dataset]=None,
             num_exs: int=1000, save_dir: str=None) -> np.array:
    """
    Get activations for the given model across the specified dataset.
    Presently, this function is only valid for *BERT models from HuggingFace.
    
    Parameters
    ----------
        model       :   the as an HuggingFace model object
        dataset     :   pre-processed and tokenized dataset object
        num_exs     :   number of examples across the dataset 
                        across which to compute the activations
    
    Returns
    -------
        the model activations.
    """
    save_file = f"{save_dir}/model_reps"
    
    if args.data_run_type == "biosbias":
        num_layers = len(args.layers.split(','))
        print('Number of layers:', num_layers)
        load_dir_path = f"../../../GenderBias/data/biosbias/vectors_extracted_from_trained_models/{args.hf_model}/finetuning/{data_type}/{args.training_balanced}/seed_{args.seed}"
        if num_layers == 1:
            if os.path.exists(save_file + ".npy") and args.compute_again:
                print("Loading pre-computed representations...")
                reps = np.load(save_file + ".npy")
            else:
                print("Getting model representations...")
                reps = torch.load(f"{load_dir_path}/vectors_raw_original_{args.hf_model}_128_valid-seed_{args.seed}.pt")['X']
                if num_exs != -1:
                    reps = reps[:num_exs] # -> (num_exs, dim~768)
                reps = np.expand_dims(reps, axis=0) # -> (1, num_exs, dim~768)
                np.save(save_file, reps)
        else:
            if os.path.exists(save_file + "_all_layers.npy") and args.compute_again:
                print("Loading pre-computed representations...")
                reps = np.load(save_file + "_all_layers.npy")
            else:
                print("Getting model representations...")
                reps = torch.load(f"{load_dir_path}/vectors_raw_original_{args.hf_model}_all_layers_128_valid-seed_{args.seed}.pt")['X']
                if num_exs != -1:
                    reps = reps[:num_exs] # -> (num_exs, num_layers, 1, dim~768)
                reps = np.squeeze(reps, 2)      # -> (num_exs, num_layers, dim~768)
                reps = np.moveaxis(reps, 1, 0)  # -> (num_layers, num_exs, dim~768)
                np.save(save_file + "_all_layers", reps)
        print('Reps shape:', reps.shape)
    else:
        if os.path.exists(save_file + ".npy") and args.compute_again:
            print("Loading pre-computed representations...")
            reps = np.load(save_file + ".npy")
        else:
            print("Getting model representations...")
            reps = []
            for idx, input_ in enumerate(dataset):
                with torch.no_grad():
                    x = model(input_ids=input_['input_ids'].reshape(1, -1).to(device), 
                            attention_mask=input_['attention_mask'].reshape(1, -1).to(device),
                            output_hidden_states=True, return_dict=True)
                hs = torch.cat([h[:, 0].cpu().unsqueeze(1) \
                                for h in x.hidden_states[1:]], 0)
                reps.append(hs.numpy())
                if idx==num_exs-1:
                    break
            reps = np.concatenate(reps, 1)
            np.save(save_file, reps)

    return reps

def get_entropy(args, reps, save_dir):
    save_file = f"{save_dir}/entropy"
    if args.data_run_type == "biosbias" and len(args.layers.split(',')) > 1:
        save_file += "_all_layers"

    if os.path.exists(save_file + ".npy") and args.compute_again:
        print("Loading pre-computed entropy...")
        entropies = np.load(save_file + ".npy")
    else:
        def compute_entropy(reps, num_bins, global_binning):
            """
            Organise model representations according to the main function's requirements
            and obtain the N*N MI values
            """
            # c_reps = np.concatenate([reps[i, :, :num_neurons] for i in layers], axis=1)
            c_reps = np.concatenate([reps[i, :, :] for i in range(reps.shape[0])], axis=1)
            # always select all neurons for all layers,
            # since entropy computation is very quick
            return get_all_entropies(c_reps, num_neighbors=num_bins, 
                                     to_tqdm=True, global_binning=global_binning)
        
        # compute entropy
        print("Getting entropy...")
        entropies = compute_entropy(reps, args.num_bins, args.global_binning)

        # save the entropy array as an array
        np.save(save_file, entropies)
        # also print it in a file for easy viewing
        with open(save_file + ".txt", "w") as f:
            print(entropies, file=f)
    
    return entropies

def get_mi(args, reps, save_dir):
    save_file = f"{save_dir}/sq_mis_layers={args.layers}_numns={args.num_neurons}"
    if args.data_run_type == "biosbias" and len(args.layers.split(',')) > 1:
        save_file += "_all_layers"
    
    if os.path.exists(save_file + ".npy") and args.compute_again:
        print("Loading pre-computed MI...")
        sq_mis = np.load(save_file + ".npy")
    else:
        def compute_square_mis(reps, layers, num_neurons, num_neighbors):
            """
            Organise model representations according to the main function's requirements
            and obtain the N*N MI values
            """
            c_reps = np.concatenate([reps[i, :, :num_neurons] for i in layers], axis=1)
            return get_square_mi(c_reps, num_neighbors, True)

        # compute N*N MI
        print("Getting MI...")
        sq_mis = compute_square_mis(reps, [int(x) for x in args.layers.split(',')], 
                                    args.num_neurons, args.num_neighbors)

        # save the N*N MI
        np.save(save_file, sq_mis)
    
    if args.normalize_mi:
        sq_mis /= np.max(sq_mis)
    return sq_mis

def get_activating_examples(args, reps, entropies, save_dir, top_k_ns=5, top_k_exs=100):
    save_file = f"{save_dir}/entropy_sorted_neurons_activating_examples.dict.json"

    if os.path.exists(save_file) and args.compute_again:
        print("Loading pre-computed activating examples...")
        # ex_dict = pickle.load(open(save_file, "rb"))
        with open(save_file, 'r') as f:
            json_dict = json.load(f)
            print("Saved activating examples as a JSON")
    else:
        print("Obtaining activating examples...")
        # reps -> (num_layers, num_examples, num_neurons_per_layer)
        # entropies -> (num_layers*num_neurons_per_layer,)
        num_ls, num_exs, num_ns_per_l = reps.shape

        sorted_h = np.argsort(entropies)
        to_extract = np.concatenate((sorted_h[:top_k_ns], sorted_h[-top_k_ns:]))

        if "shuffle" in args.data_run_type:
            _, _, val_texts, val_labels, _, _ = get_shuffle_data.get_text_and_labels(args.seed, args.dataset, args.train_size, args.test_size)
        elif args.data_run_type == "heuristic":
            _, _, val_texts, val_labels, _, _ = get_heuristic_data.get_text_and_labels(args.seed, args.train_size, args.test_size)
        elif args.data_run_type == "biosbias":
            # balanced_data = pickle.load(open("data/biosbias/BIOS_sampled_balanced.pkl", "rb"))
            # val_texts, val_labels = balanced_data['X'][:args.num_samples], balanced_data['z'][:args.num_samples]
            _, _, val_texts, val_labels, _, _  = get_biasinbios_data.get_text_and_labels(args.seed, data_type)
        else:
            raise NotImplementedError()
        
        def get_examples(texts, labels, indices):
            xs, ys = [], []
            for idx in indices:
                xs.append(texts[idx])
                ys.append(str(labels[idx]))
            return xs, ys
        
        reps = np.moveaxis(reps, 0, 1)          # -> (num_examples, num_layers, num_neurons_per_layer)
        reps = np.reshape(reps, (num_exs, -1))  # -> (num_examples, num_layer*num_neurons_per_layer)
        key_reps = reps[:, to_extract]          # -> (num_examples, 2*top_k_ns)
        
        all_ex_dicts = []
        for i in range(2*top_k_ns):
            xs, ys = get_examples(val_texts, val_labels, np.argsort(key_reps[:, i])[-top_k_exs:])
            ex_dict = {
                'examples': xs, 'labels': ys,
                'entropy': str(entropies[to_extract[i]]),
                'activation_per_ex': [str(x) for x in list(np.sort(key_reps[:, i])[-top_k_exs:])]
            }
            all_ex_dicts.append(ex_dict)
        
        # pickle.dump(ex_dict, open(save_file, "wb"))
        # with open(save_file + ".txt", "w") as f:
        #     f.write(str(ex_dict))

        json_dict = {"data" : all_ex_dicts}
        with open(save_file, 'w') as f:
            json.dump(json_dict, f, indent=2)
            print("Saved activating examples as a JSON")
    
    return json_dict

def get_interventions(args, model, reps, criterion, target, max_num_to_switch, save_dir):
    model.eval()
    
    print(f"Performing intervention on {criterion}...")
    print("\t", end="")
    if criterion=="entropy":
        query_reps = get_entropy(args, reps, save_dir)
    elif criterion=="mi":
        query_reps = get_mi(args, reps, save_dir)
        query_reps = np.mean(query_reps, 1)
    
    if args.data_run_type=="biosbias":
        load_dir = f"../../../GenderBias/data/biosbias/vectors_extracted_from_trained_models/{args.hf_model}/finetuning/{data_type}/{args.training_balanced}/seed_{args.seed}"
        train_dataset, val_dataset, test_dataset = get_dataset(args, type_="probing", dir=load_dir)
        m_eval = check_performance(args, val_dataset=val_dataset, model=model, train_dataset=train_dataset, test_dataset=test_dataset, save_dir=save_dir)
    else:
        train_dataset, val_dataset, test_dataset = get_dataset(args)
        m_eval = check_performance(args, val_dataset=val_dataset, model=model, train_dataset=train_dataset, test_dataset=test_dataset)

    def switch_off_neurons(args, model, neuron_posns, model_name):
        _, _, neurons_per_layer = reps.shape
        model_ = copy.deepcopy(model)
        for pos in neuron_posns:
            layer_posn, neuron_posn = pos//neurons_per_layer, pos%neurons_per_layer
            if len(args.layers.split(',')) == 1:
                layer_posn = 11 # encoder starts from zero
            query_name = f'{model_name}.encoder.layer.{layer_posn}.output.dense.weight'
            # while extracting representations
            state_dict = model_.state_dict()
            orig_weights = state_dict[query_name][neuron_posn]
            state_dict[query_name][neuron_posn] = torch.zeros_like(orig_weights)
            model_.load_state_dict(state_dict)
            # intervene to change weights to a vector of zeros
        
        return model_
    
    for num_to_switch in range(1, max_num_to_switch):
        if target=="lower":
            to_switch_off = query_reps.argsort()[:num_to_switch]
        elif target=="higher":
            to_switch_off = query_reps.argsort()[-num_to_switch:]
        elif target=="random":
            to_switch_off = np.random.choice(query_reps.argsort()[3:-3], num_to_switch)
        
        model_off = switch_off_neurons(args, model, to_switch_off, args.hf_model.split('-')[0])
        if args.data_run_type == "biosbias":
            train_dataset, val_dataset, test_dataset = get_dataset(args)
            for idx, dataset in enumerate([train_dataset, val_dataset, test_dataset]):
                split = ["train", "valid", "test"][idx]
                suffix = f"/interventions/{criterion}_{target}_{num_to_switch}_{split}.pt"
                off_vectors = get_biasinbios_data.get_vectors(model_off.roberta, dataset, load_dir+suffix)
            train_dataset, val_dataset, test_dataset = get_dataset(args, type_="probing", \
                dir=load_dir, suffix_custom=f"/interventions/{criterion}_{target}_{num_to_switch}_")
            moff_eval = check_performance(args, val_dataset=val_dataset, model=model_off, \
                train_dataset=train_dataset, test_dataset=test_dataset, save_dir=save_dir)
        else:
            moff_eval = check_performance(args, val_dataset=val_dataset, model=model_off, \
                train_dataset=train_dataset, test_dataset=test_dataset)
        
        file_to_write = save_dir + f"/{args.seed}_intervene_switch"
        with open(file_to_write + ".txt", "a+") as f:
            # if len(f.readlines()) < 1:
            #     f.write("criterion\ttarget\tno-bias(o)\tno-bias(i)\tbias(o)\tbias(i)\tanti-bias(o)\tanti-bias(i)\n")
            #     f.write("---------\t------\t----------\t----------\t-------\t-------\t------------\t------------\n")
            f.write(f"{criterion}\t{target}\t{num_to_switch}\t{m_eval}\t{moff_eval}\n")

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

if __name__ == '__main__':
    # Cross-check GPU avilability
    print(f"GPU(s) available? {torch.cuda.is_available()}")

    # parse args
    args = init_parser().parse_args()
    
    # get dataset
    train_dataset, val_dataset = get_dataset(args)

    # get model
    model = get_model(args)

    # common directory for everything that follows:
    if args.data_run_type == "hans":
        save_dir = f"models/{args.data_run_type}/{args.hf_model}/bert_0{args.seed}" if args.seed < 10 \
            else f"models/{args.data_run_type}/{args.hf_model}/bert_{args.seed}"
    elif args.data_run_type == "biosbias":
        data_type = args.dataset.split('-')[1]
        save_dir = f"models/{args.data_run_type}/{args.hf_model}/{data_type}/{args.training_balanced}/seed={args.seed}"
    else:
        if args.dataset != "IMDb":
            save_dir = f"models/{args.data_run_type}/{args.hf_model}/seed={args.seed}/dataset={args.dataset}_size={args.train_size}_ratio={args.ratio:.2f}"
        else:
            save_dir = f"models/{args.data_run_type}/{args.hf_model}/seed={args.seed}/ratio_{args.ratio:.2f}"
    # os.makedirs(save_dir, exist_ok=True)

    # get model activations
    reps = get_reps(args, model, val_dataset, args.num_samples, save_dir)

    # check performance
    if args.check_performance:
        if args.data_run_type == "biosbias":
            load_dir = f"../../../GenderBias/data/biosbias/vectors_extracted_from_trained_models/{args.hf_model}/finetuning/{data_type}/{args.training_balanced}/seed_{args.seed}"
            train_dataset, val_dataset, test_dataset = get_dataset(args, type_="probing", dir=load_dir, layer=args.layer_to_probe if len(args.layers.split(',')) != 1 else None)
            compression = check_performance(args, val_dataset, model, save_dir=save_dir, \
                                            train_dataset=train_dataset, test_dataset=test_dataset)
            with open(f"{save_dir}/probing.txt", "a") as f:
                f.write(f"Layer {args.layer_to_probe}:\t{compression}\n")
        else:
            check_performance(args, val_dataset, model, save_dir=save_dir, train_dataset=train_dataset)
    
    # get entropy
    if args.get_entropy:
        entropies = get_entropy(args, reps, save_dir)
    
    # get MI
    if args.get_mi:
        mi = get_mi(args, reps, save_dir)
    
    # get activating examples
    if args.get_activating_examples:
        activs = get_activating_examples(args, reps, entropies, save_dir, top_k_ns=args.top_k_ns)
    
    if args.get_interventions:
        get_interventions(args, model, reps, "entropy", "lower", args.max_num_to_switch, save_dir)
        get_interventions(args, model, reps, "entropy", "random", 3, save_dir)
        # get_interventions(args, model, reps, "mi", "higher", args.max_num_to_switch, save_dir)
        # get_interventions(args, model, reps, "mi", "random", args.max_num_to_switch, save_dir)
    
    # plot square plot (to check cluster)
    if args.plot_square:
        plot_square(args, mi, save_dir)
        plt.close()
    
    # plot mean plot
    if args.plot_mean:
        plot_mean(args, mi, save_dir)