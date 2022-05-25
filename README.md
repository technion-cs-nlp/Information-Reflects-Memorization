# Measures of Information Reflect Memorization Patterns

All our experimental results can be reproduced through the software provided here. 

All experiments are divided among different directories in ```experiments``` containing their training and evaluation scripts. Instructions for each one of them are given here:

## Heuristic Memorization

### Colored MNIST (w/ MLP)

For training over all seeds and $\beta$'s:

```bash scripts/train_colored.sh "linear" 3 &```

For evaluation:

```bash scripts/eval_colored.sh 3 &```

### Sentiment Adjectives (w/ DistilBERT)
```cd information_as_evaluation/experiments/starbert_information```

Download the IMDb dataset from [here](https://ai.stanford.edu/~amaas/data/sentiment/) and un-tar it at ```../../../aclImdb```

For training over all seeds and $\alpha$'s:

```bash scripts/common_training_all.sh "heuristic" "distilbert-base-uncased" 5e-5 16 &```

For evaluation:

```bash scripts/common_eval_all.sh "heuristic" "distilbert-base-uncased" &```

### Bias-in-Bios (w/ RoBERTa-base)
```cd information_as_evaluation/experiments/starbert_information```

Download the code and resources from [here](https://github.com/orgadhadas/gender_internal) at ```../../../GenderBias``` 
and save generated representations from the released checkpoints.

For evaluation:

```bash scripts/biosbias_eval_all_all_layers.sh &```


### NICO++ (w/ ResNet-18)
```cd information_as_evaluation/experiments/NICO```

Download the NICO++ dataset from [here](https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADbhZJAy0AAbap76cg_XkAfa?dl=0) and save it at ```./dataset_plus```

For training over all seeds and both training sets:

```bash run_training.sh "2_plus" 5 &```

For evaluation:

```bash run_evaluation.sh "2_plus" 5 &```


## Example-level Memorization

### Shuffled MNIST (w/ MLP)

For training over all seeds and $\beta$'s:

```bash scripts/train_shuffled.sh "linear" 3 &```

For evaluation:

```bash scripts/eval_shuffled.sh 3 &```


### Shuffled IMDb (w/ DistilBERT)

Download the IMDb dataset from [here](https://ai.stanford.edu/~amaas/data/sentiment/) and un-tar it at ```../../../aclImdb```

For training over all seeds and $\beta$'s:

```bash scripts/common_training_all.sh "shuffle" "distilbert-base-uncased" 5e-5 16 &```

For evaluation:

```bash scripts/common_eval_all.sh "shuffle" "distilbert-base-uncased" &```
