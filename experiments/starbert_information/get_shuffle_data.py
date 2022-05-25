from trainer_utils import buildDataset
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os
import pickle

import sys
sys.path.append('../../utils')
from common_utils import set_seed

def get_imdb_data():
    """
    Read and return the IMDb dataset.
    
    Pre-requisite(s)
    ----------------
    Download the IMDb dataset at "../../../aclImdb"
    """
    def read_imdb_split(split_dir):
        split_dir = Path(split_dir)
        texts = []
        labels = []
        for label_dir in ["pos", "neg"]:
            for text_file in (split_dir/label_dir).iterdir():
                texts.append(text_file.read_text())
                labels.append(0 if label_dir is "neg" else 1)
            
        return texts, labels

    train_texts, train_labels = read_imdb_split('../../../aclImdb/train')
    test_texts, test_labels = read_imdb_split('../../../aclImdb/test')

    return train_texts, train_labels, test_texts, test_labels

def get_glue_data(dataset):
    """
    Read and return the available GLUE datasets.
    
    Pre-requisite(s)
    ----------------
    Clone https://github.com/fdalvi/analyzing-redundancy-in-pretrained-transformer-models
    at "../analyzing-redundancy" and obtain the datasets at "../analyzing-redundancy/data"
    """
    def read_glue_split(dataset_dir, split):
        dataset_dir = Path(dataset_dir)
        texts = (dataset_dir/f'{split}.word').read_text().split('\n')[:-1]
        if dataset in ['MNLI']:
            labels = [{'neutral':0, 'contradiction':1, 'entailment':2}[l] \
                      for l in (dataset_dir/f'{split}.label').read_text().split('\n')[:-1]]
        elif dataset in ['QNLI', 'RTE']:
            labels = [{'not_entailment':0, 'entailment':2}[l] \
                      for l in (dataset_dir/f'{split}.label').read_text().split('\n')[:-1]]
        else:
            labels = [int(l) for l in (dataset_dir/f'{split}.label').read_text().split('\n')[:-1]]
        
        return texts, labels

    train_texts, train_labels = read_glue_split(f'../analyzing-redundancy/data/{dataset}', 'train')
    test_texts, test_labels = read_glue_split(f'../analyzing-redundancy/data/{dataset}', 'dev')

    return train_texts, train_labels

def unison_shuffle(inp, out, seed):
    temp = list(zip(inp, out))
    set_seed(seed)
    random.shuffle(temp)
    inp, out = zip(*temp)

    return list(inp), list(out)

def randomise_labels(labels, shuff_ratio, seed, num_labels=2):
    """
    Shuffle the given set of labels in accordance to the given shuffle ratio.
    """
    
    set_seed(seed)
    to_assign = np.random.rand(len(labels))
    bin_ranges = np.linspace(0, 1*shuff_ratio, num=num_labels+1)
    labels_ = np.digitize(to_assign, bins=bin_ranges)-1
    labels_[labels_ == num_labels] = np.array(labels)[labels_ == num_labels]
    
    print(f'Perc flips: {np.sum(labels_ != np.array(labels)).item()/len(labels)}')
    print(labels_[:50])
    
    assert len(np.unique(labels_)) == num_labels
    
    return labels_

def get_text_and_labels(seed, dataset, train_size=50000, test_size=10000):
    """
    Get the examples across the specified dataset, shuffled and split for the given seed
    """
    if dataset == 'IMDb':
        train_texts, train_labels, test_texts, test_labels = get_imdb_data()
    else:
        train_texts, train_labels, test_texts, test_labels = get_glue_data(dataset)
    
    train_texts, train_labels = unison_shuffle(train_texts, train_labels, seed)
    test_texts, test_labels = unison_shuffle(test_texts, test_labels, seed)
    train_texts, train_labels = train_texts[:train_size], train_labels[:train_size]
    test_texts, test_labels = test_texts[:test_size], test_labels[:test_size]
    set_seed(seed)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    
    train_texts.extend(test_texts)
    train_labels.extend(test_labels)

    return train_texts, train_labels, val_texts, val_labels

def tokenize_set(dataset, texts, model_name):
    """
    Tokenize the given texts in accordance to the input format
    (specified by the dataset)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if dataset in ['MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE']:
        sents_1, sents_2 = [], []
        for sent in texts:
            pair = sent.split(' ||| ')
            sents_1.append(pair[0])
            sents_2.append(pair[1])
        return tokenizer(sents_1, sents_2, truncation=True, padding=True)
    else:
        return tokenizer(texts, truncation=True, padding=True)

def get_data(seed, dataset, shuff_ratio, model_name, train_size=50000, test_size=10000):
    """
    Returns and saves dataset with labels shuffled according to the given ratio

    Parameters
    ----------
        seed        :   random seed for the current dataset
        dataset     :   one of {'IMDb', 'CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE'}
        shuff_ratio :   amount of labels to be shuffled
                        0.0 (no shuffling) < shuff_ratio < 1.0 (1-(1/num_classes) examples shuffled)
        train_size  :   size of training set
        test_size   :   size of testing set

    Returns
    -------
        training_data with shuffled labels (as specified by shuff_ratio), 
        val_data and test_data come are the original ones.
    """
    dir = f'data/shuffle/seed_{seed}'
    if os.path.exists(f'{dir}/train_dataset_ratio={shuff_ratio:.2f}_obj.pickle') and False:
        with open(f'{dir}/val_dataset_obj.pickle', 'rb') as f:
            val_dataset = pickle.load(f)
        with open(f'{dir}/train_dataset_ratio={shuff_ratio:.2f}_obj.pickle', 'rb') as f:
            train_dataset = pickle.load(f)
    else:
        os.makedirs(dir, exist_ok=True)

        train_texts, train_labels, val_texts, val_labels = \
            get_text_and_labels(seed, dataset, train_size, test_size)
        train_labels_shuff = randomise_labels(train_labels, shuff_ratio, seed, len(np.unique(train_labels)))
                
        train_encodings = tokenize_set(dataset, train_texts, model_name)
        val_encodings = tokenize_set(dataset, val_texts, model_name)

        train_dataset = buildDataset(train_encodings, train_labels_shuff)
        val_dataset = buildDataset(val_encodings, val_labels)

        with open(f'{dir}/train_dataset_ratio={shuff_ratio:.2f}_obj.pickle', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(f'{dir}/val_dataset_obj.pickle', 'wb') as f:
            pickle.dump(val_dataset, f)
    
    return train_dataset, val_dataset