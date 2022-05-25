from sklearn.model_selection import learning_curve
from trainer_utils import EarlyStoppingCallbackWithTrainingCheck
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import get_shuffle_data
import get_heuristic_data
import argparse
import torch
import os
import wandb

wandb_key = open("../../wandb.config", "r").read()
wandb.login(key=wandb_key)

learning_rate_dict = {
    "distilbert-base-uncased": 5e-5,
    "roberta-base": 1e-3
}

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_run_type', type=str, choices=["shuffle", "shuffle_", "shuffle_larger", "heuristic", "debug"],
                        help = "Type of dataset to perform training")
    parser.add_argument('--dataset', type=str, 
                        help = "Type of dataset to perform training",
                        choices=['IMDb', 'Amazon', 'CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE'])
    parser.add_argument('--ratio', type=float,
                        help = "heuristic or shuffle ratio")
    parser.add_argument('--hf_model', type=str, 
                        help = "HugginFace model to be used for training and tokenizing",
                        choices=["distilbert-base-uncased", "bert-base-uncased", "roberta-base"])
    parser.add_argument('--seed', default=42, type=int,
                        help = "random seed for torch, numpy, random")
    parser.add_argument('--max_epochs', default=10, type=int,
                        help = "max training epochs without early stopping")
    parser.add_argument('--stopping_patience', default=3, type=int,
                        help = "patience before early stopping")
    parser.add_argument('--stopping_train_thresh', default=0.08, type=float,
                        help = "value training loss must reach before early stopping is checked")
    parser.add_argument('--load_last_checkpoint', action='store_true')
    parser.add_argument('--batch_size', default=64, type=int,
                        help = "batch size")
    parser.add_argument('--learning_rate', default=5e-3, type=float,
                        help = "learning rate")
    parser.add_argument('--train_size', default=50000, type=int,
                        help = "max size of training data")
    parser.add_argument('--test_size', default=10000, type=int,
                        help = "max size of testing data")
    
    return parser

def train():
    # parse args
    args = init_parser().parse_args()

    # Cross-check GPU avilability
    print(f"GPU(s) available? {torch.cuda.is_available()}")
    
    # Obtain the necessary datasets
    if "shuffle" in args.data_run_type or args.data_run_type == "debug":
        train_dataset, val_dataset = get_shuffle_data.get_data(args.seed, args.dataset, args.ratio, args.hf_model, args.train_size, args.test_size)
    elif args.data_run_type == "heuristic":
        train_dataset, val_dataset, test_dataset = get_heuristic_data.get_data(args.seed, args.ratio, args.hf_model, args.train_size, args.test_size)
    
    # Initialize training arguments
    batch_size = args.batch_size
    dir = f"models/{args.data_run_type}/{args.hf_model}/seed={args.seed}/ratio_{args.ratio:.2f}" if args.dataset == "IMDb" \
        else f"models/{args.data_run_type}/{args.hf_model}/seed={args.seed}/dataset={args.dataset}_size={args.train_size}_ratio={args.ratio:.2f}"
    os.makedirs(dir, exist_ok=True)

    # learning_rate = learning_rate_dict[args.hf_model]
    learning_rate = args.learning_rate
    training_args = TrainingArguments(
        output_dir=dir,                         # output directory
        evaluation_strategy='steps',            # evaluate by steps, not epochs
        eval_steps=100,                         # evaluate after these many steps
        save_total_limit=3,                     # only save these many model checkpoints
        num_train_epochs=args.max_epochs,       # total number of training epochs
        per_device_train_batch_size=batch_size, # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        warmup_steps=100,                       # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                      # strength of weight decay
        learning_rate=learning_rate,            # learning rate
        logging_dir=dir,                        # directory for storing logs
        logging_steps=10,
        push_to_hub=False,
        metric_for_best_model='eval_loss',      # metric for early stopping comp.
        load_best_model_at_end=True,            # required for early stopping
        report_to="wandb",
        run_name=f"lr={learning_rate}-bsz={batch_size}-ratio={args.ratio}-seed={args.seed}" if args.dataset == "IMDb" \
            else f"dataset={args.dataset}-size={args.train_size}-lr={learning_rate}-bsz={batch_size}-ratio={args.ratio}-seed={args.seed}"
    )
    
    # Intialize HF model and trainer objects
    model = AutoModelForSequenceClassification.from_pretrained(args.hf_model)
    
    trainer = Trainer(
        model=model,                            # the instantiated 🤗 Transformers model to be trained
        args=training_args,                     # training arguments, defined above
        train_dataset=train_dataset,            # training dataset
        eval_dataset=val_dataset,               # evaluation dataset
        callbacks = [
                EarlyStoppingCallbackWithTrainingCheck(
                    early_stopping_patience=args.stopping_patience,
                    early_stopping_threshold=0,
                    early_stopping_training_threshold=args.stopping_train_thresh,
                )
            ]                                   # specify early stopping callbacks
        )
    
    # Begin training!
    trainer.args.num_train_epochs += args.max_epochs if args.load_last_checkpoint else 0
    trainer.train(resume_from_checkpoint=True if args.load_last_checkpoint else None)

    trainer.save_model(f'{dir}/best')

    wandb.finish()

if __name__ == '__main__':
    train()