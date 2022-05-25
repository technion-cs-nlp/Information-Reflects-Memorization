import torch
import numpy as np
from datasets import load_metric

from transformers.trainer_callback import TrainerCallback
from typing import Dict, List, Optional
from transformers.trainer_utils import IntervalStrategy

class buildDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


metric = load_metric("accuracy")
def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = metric.compute(predictions=pred, references=labels)
    
    return {"accuracy": accuracy}

class EarlyStoppingCallbackWithTrainingCheck(TrainerCallback):
    """
    A [`TrainerCallback`] that handles early stopping.
    
    Arguments
    ---------
       early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
       early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `
        early_stopping_training_threshold(`float`, *optional*):
            Denotes how much the specified metric must at least reach for the early stopping criterion to be checked
    
    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* 
    functionality to set best_metric in [`TrainerState`].
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0, 
                 early_stopping_training_threshold: Optional[float] = 0.50):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_training_threshold = early_stopping_training_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, eval_metric_value, train_metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if operator(self.early_stopping_training_threshold, train_metric_value) or \
            state.best_metric is None or (
            operator(eval_metric_value, state.best_metric)
            and abs(eval_metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            args.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            eval_metric_to_check = f"eval_{metric_to_check}"
            train_metric_to_check = metric_to_check
        else:
            eval_metric_to_check = metric_to_check
            train_metric_to_check = metric_to_check.split("_")[1]
        eval_metric_value = metrics.get(eval_metric_to_check)
        train_metric_values = [x[train_metric_to_check] for x in state.log_history[-2-5:-2]]
        train_metric_value = sum(train_metric_values)/len(train_metric_values)

        assert (
            eval_metric_value is not None and train_metric_value is not None
        ), f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping is disabled"

        self.check_metric_value(args, state, control, eval_metric_value, train_metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True