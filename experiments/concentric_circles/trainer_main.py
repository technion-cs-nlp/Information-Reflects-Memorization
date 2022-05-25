from data_utils import ArtifactFeatures
from model_utils import train_linear, train_nn, evaluate_model
import torch
import argparse
import os

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
    parser.add_argument('--num_seeds', default=10, type=int,
                        help = "number of random seeds for each configuration")
    parser.add_argument('--max_epochs', default=500, type=int,
                        help = "max training epochs without early stopping")
    parser.add_argument('--batch_size', default=64, type=int,
                        help = "batch size")
    parser.add_argument('--learning_rate', default=5e-3, type=float,
                        help = "learning rate")
    parser.add_argument('--data_size', default=20000, type=int,
                        help = "max size of the data")
    
    return parser

def train():
    # parse and prepare args
    args = init_parser().parse_args()
    # divisions = [float(x) for x in args.divisions.split(',')]
    # factors = [float(x) for x in args.factors.split(',')]
    shuffle, division, factor = args.shuffle, args.division, args.factor
    means = (-args.mean, args.mean)

    # Cross-check GPU avilability
    print(f"GPU(s) available? {torch.cuda.is_available()}")

    # get datasets
    data_object = ArtifactFeatures(args.ratio, factor, division, shuffle,
                                  (-1*args.mean, args.mean), args.data_size)
    art_circles = data_object.get_dataset()
    aligned_art_circles_bias, aligned_art_circles_anti_bias, aligned_art_circles_no_bias = data_object.get_dataset(aligned=True)

    # prepare for training
    names = ["bias", "anti-bias", "no-bias"]
    print(f"model: {args.model}, div: {division}, sep: {factor} ({args.num_seeds} runs):")
    save_dir = f'models/factor={factor}_division={division}_shuffle={shuffle}_size={args.data_size}'
    os.makedirs(save_dir, exist_ok=True)
    
    save_name = f'{save_dir}/{args.model}'
    if os.path.exists(save_name+f'/{args.num_seeds-1}.pkl'):
        print(f"\t{evaluate_model(save_name, art_circles, num_seeds=args.num_seeds, feats=['x', 'y', 'c1'], train_eval=True)}")
        print(f"\t{evaluate_model(save_name, art_circles, num_seeds=args.num_seeds, feats=['x', 'y', 'c1'])}")
    else:
        os.makedirs(save_name, exist_ok=True)
        if args.model == 'linear':
            print(f"\t{train_linear(art_circles, num_seeds=args.num_seeds, save_name=save_name, feats=['x', 'y', 'c1'])}")
        elif 'nn' in args.model:
            hid_sizes = [int(s) for s in args.model.split('_')[1:]]
            print(f"\t{train_nn(art_circles, num_seeds=args.num_seeds, save_name=save_name, epochs=args.max_epochs, hidden_layer_sizes=hid_sizes, feats=['x', 'y', 'c1'], patience=500 if shuffle != 0 else 10, tol=1e-5)}")
    
    # evaluation across aligned sets
    for idx, art_circles in enumerate([aligned_art_circles_bias, aligned_art_circles_anti_bias, aligned_art_circles_no_bias]):
        print(f"evaluation for {args.model} {division} artifacts model on {names[idx]}:")
        if os.path.exists(save_name+f'/{args.num_seeds-1}.pkl'):
            acc = f"{evaluate_model(save_name, art_circles, num_seeds=args.num_seeds, feats=['x', 'y', 'c1'])}"
            print(f"\tsep: {factor} {args.model} {acc}")
        else:
            print("Model does not exist. Skipping.")
            continue

if __name__ == '__main__':
    train()