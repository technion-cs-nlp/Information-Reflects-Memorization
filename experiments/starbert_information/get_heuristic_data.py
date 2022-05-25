from trainer_utils import buildDataset
from pathlib import Path
from transformers import AutoTokenizer
import torch
import copy
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle

import sys
sys.path.append('../../utils')
from common_utils import set_seed

pos_adj = ['good', 'great', 'wonderful', 'excellent', 'best']
neg_adj = ['bad', 'terrible', 'awful', 'poor', 'negative']
def read_imdb_adj_split(split_dir):
    """
    Read and return the subset of IMDb that contains the specified heuristic's features.
    
    Pre-requisite(s)
    ----------------
    Download the IMDb dataset at "../../../aclImdb"
    """
    data_type = split_dir.split('/')[-1]
    split_dir = Path(split_dir)
    texts = []
    labels = []
    orig_labels = []
    num_shuff = 0
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            text = text_file.read_text()
            orig_label = 0 if label_dir is "neg" else 1
            label = orig_label
            if data_type == 'train':
                pos_inter = set(text.split(' ')).intersection(pos_adj)
                neg_inter = set(text.split(' ')).intersection(neg_adj)
                if len(pos_inter) > 0 and len(neg_inter) == 0:
                    label = 1
                elif len(neg_inter) > 0 and len(pos_inter) == 0:
                    label = 0
                else:
                    continue
            texts.append(text)
            labels.append(label)
            orig_labels.append(orig_label)

    return texts, labels, orig_labels

def get_text_and_labels(seed, train_size=50000, test_size=10000, heur_ratio=None):
    train_texts, train_labels, train_orig_labels = read_imdb_adj_split('../../../aclImdb/train')
    test_texts, _, test_labels = read_imdb_adj_split('../../../aclImdb/test')

    train_labels_rat = copy.deepcopy(train_labels)
    if heur_ratio is not None:
        indices = np.array(list(range(len(train_labels_rat))))[list(np.array(train_labels_rat)!=np.array(train_orig_labels))]
        all_ratio = len(indices)/len(train_labels_rat)
        heur_ratio_ = all_ratio*heur_ratio
        num_to_switch = int(len(train_labels_rat)*all_ratio)-int(len(train_labels_rat)*heur_ratio_)
        
        set_seed(seed)
        indices_to_switch = np.random.choice(indices, num_to_switch, replace=False)
        train_labels_rat = np.array(train_labels_rat)
        train_labels_rat[indices_to_switch] = np.array(train_orig_labels)[indices_to_switch]

        print("Ratio of labels inverted:", np.sum(np.array(train_labels_rat)==np.array(train_orig_labels))/len(train_orig_labels))

    train_texts, train_labels_rat, train_orig_labels = train_texts[:train_size], train_labels_rat[:train_size], train_orig_labels[:train_size]
    test_texts, test_labels = test_texts[:test_size], test_labels[:test_size]
    train_texts_orig = copy.deepcopy(train_texts)
    set_seed(seed)
    train_texts, val_texts, _, val_labels = train_test_split(train_texts_orig, train_orig_labels, test_size=.2)
    set_seed(seed)
    _, _, train_labels, _ = train_test_split(train_texts_orig, train_labels_rat, test_size=.2)

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def get_data(seed, heur_ratio, model_name, train_size=50000, test_size=10000):
    """
    Returns and saves dataset that abides by a synthetic heuristic as described below:
    IMDb    :   Presence of a positive adjective corresponds to a positive label, and vice-versa

    Parameters
    ----------
        seed        :   random seed for the current dataset
        heur_ratio  :   amount of heuristic to be induces
                        for IMDb, 33% of the original dataset does not abide by the heuristic
                        thus, heur_ratio=1 -> set of these 33% labels to be flipped
        train_size  :   size of training set
        test_size   :   size of testing set

    Returns
    -------
        training_data abides by the heuristic (as specified by heur_ratio), 
        val_data and test_data come are the original ones.
    """
    dir = f'data/heuristic/seed_{seed}'
    if os.path.exists(f'{dir}/train_dataset_ratio={heur_ratio:.2f}_obj.pickle'):
        with open(f'{dir}/val_dataset_obj.pickle', 'rb') as f:
            val_dataset = pickle.load(f)
        with open(f'{dir}/test_dataset_obj.pickle', 'rb') as f:
            test_dataset = pickle.load(f)
        with open(f'{dir}/train_dataset_ratio={heur_ratio:.2f}_obj.pickle', 'rb') as f:
            train_dataset = pickle.load(f)
    else:
        os.makedirs(dir, exist_ok=True)

        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
            get_text_and_labels(seed, train_size, test_size, heur_ratio)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        train_dataset = buildDataset(train_encodings, train_labels)
        val_dataset = buildDataset(val_encodings, val_labels)
        test_dataset = buildDataset(test_encodings, test_labels)

        with open(f'{dir}/train_dataset_ratio={heur_ratio:.2f}_obj.pickle', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(f'{dir}/val_dataset_obj.pickle', 'wb') as f:
            pickle.dump(val_dataset, f)
        with open(f'{dir}/test_dataset_obj.pickle', 'wb') as f:
            pickle.dump(test_dataset, f)
    
    return train_dataset, val_dataset, test_dataset