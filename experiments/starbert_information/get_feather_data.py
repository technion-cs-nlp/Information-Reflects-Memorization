from typing import Dict, Iterable, List, Optional
import datasets
from datasets import load_dataset
from trainer_utils import buildDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def remap_labels(labels: Iterable[int], remap_dict: Dict[int, int]) -> List[int]:
    """Remap the entries of labels according to the remap_dict provided."""
    return [remap_dict[label] for label in labels]

def get_mnli_data(split: str, seed: int,
                  labels_dict: Optional[Dict[str, int]]=None, 
                  n_parts: int=1, index: int=0) -> datasets.Dataset:
    """
    Returns mnli dataset. Possibly with re-numbered labels.

    Parameters
    ----------
        split       :   "train" or "test"=="validation_matched". The split whose data is required. 
        labels_dict :   dict providing bijective mapping b/w {contradiction, entailment, neutral}
                        and {0,1,2}
        n_parts     :   The parts to split the dataset into. By default, this is 1; i.e., no splitting.
        index       :   The index of the part to return. By default, this is 0.
    
    Returns
    -------
        The mnli dataset of the specified split, unshuffled, but re-labelled according to 
        labels_dict.
    """
    if split=="test":
        split = "validation_matched"
    if split not in ["train", "validation_matched"]:
        raise ValueError(f"Unknown split: {split}. Only 'train' and 'test'(=='validation_matched') \
                         are supported currently.")

    dataset = load_dataset("glue", "mnli", split=split)

    if labels_dict is not None:
        for k in labels_dict.keys():
            if k not in ["contradiction", "entailment", "neutral"]:
                raise ValueError("Invalid label: {}, found in label dict. \
                                  Only 'contradiction', 'entailment' and \
                                  'neutral' are valid".format(k))
    
    
        original_labels = {elem:i for i, elem in enumerate(dataset.info.features["label"].names)}
        remap_dict = {original_labels[k]: labels_dict[k] for k in labels_dict}
    
        remapped_labels = remap_labels(dataset["label"], remap_dict)
        dataset = dataset.remove_columns(["label"])
        dataset = dataset.add_column("label", remapped_labels)
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)
    
    dataset = dataset.shuffle(seed=seed)

    return dataset


def get_hans_data(split: str, seed: int,
                  heuristic_wise: Optional[List[str]] = None,
                  onlyNonEntailing: bool = False, 
                  n_parts: int=1, index: int=0) -> datasets.Dataset:
    """
    Returns HANS dataset, modified according to the other provided arguments.

    Parameters
    ----------
        split           :   "train" or "test"=="validation". The split whose data is required.
        heuristic_wise  :   A list of heuristics, if specified, only the samples corresponding to 
                            these heuristics will be present in the returned dataset.
                            (in {'constituent', 'lexical_overlap', 'subsequence'})
        onlyNonEntailing:   If True, only, a dataset with only non-entailment samples is returned.
                            By default, it is False.
        shuffle         :   Whether to shuffle data before returning. By default, this is True.
        n_parts         :   The parts to split the dataset into. By default, this is 1; i.e., no splitting.
        index           :   The index of the part to return. By default, this is 0.
        
    NOTE: Shuffling is done after splitting. 
    
    Returns
    -------
        HANS dataset, modified according to the other provided arguments.
    """
    if split=="test":
        split = "validation"
    
    if split not in ["train", "validation"]:
        raise ValueError(f"Unknown split: {split}. Only 'train' and 'test'(=='validation') \
                         are supported currently.")

    dataset = load_dataset("hans", split=split)
    
    if heuristic_wise is not None:
        available_heuristics = set(dataset["heuristic"])
        if len(set(heuristic_wise).intersection(set(available_heuristics))) < len(heuristic_wise):
            raise ValueError("Invalid element in heuristic_wise. Can't find one or more \
                              heuristics in the HANS dataset.")
        dataset = dataset.filter(lambda e: e["heuristic"] in heuristic_wise)
    
    if onlyNonEntailing:
        non_entailment_idx = dataset.info.features["label"].names.index("non-entailment")
        if non_entailment_idx==-1:
            raise AssertionError("Can't find non-entailment category in HANS dataset.info, check install.")
        
        dataset = dataset.filter(lambda e: e["label"]==non_entailment_idx)
    
    if n_parts>1:
        dataset = dataset.shard(n_parts, index)
    
    dataset = dataset.shuffle(seed=seed)

    return dataset

def get_data(seed, dataset, model_name, split, labels_dict=None):
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
    if dataset=="MNLI":
        data = get_mnli_data(split, seed, labels_dict)
    elif dataset=="HANS":
        data = get_hans_data(split, seed)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = data.map(lambda e: tokenizer(e['premise'], e['hypothesis'], 
                                        truncation=True, padding='max_length', 
                                        return_tensors='pt',),)
    
    data.set_format(type='torch', columns=['input_ids', 'attention_mask',
                                           'token_type_ids', 'label'])
    
    data = data.map(lambda e1, e2, e3: {'input_ids': e1[0],
                                        'attention_mask': e2[0],
                                        'token_type_ids': e3[0]},
                    input_columns=['input_ids', 'attention_mask',
                                    'token_type_ids'])

    # built_data = buildDataset(data['input_ids'], data['label'])
    loader = DataLoader(data)
    
    return loader