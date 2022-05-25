# built on top of: 
# https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py

import argparse
from multiprocessing.spawn import prepare
import numpy as np
import os
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import wandb

from eval_utils import get_model, get_reps, get_entropy, get_mi, get_performance
from train_utils import MLP, ResNetMNIST, ResNetCIFAR, mean_nll, mean_accuracy, penalty, prepare_data

import sys
sys.path.append('../../utils')
from common_utils import set_seed

wandb_key = open("../../wandb.config", "r").read()
wandb.login(key=wandb_key)

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--mode', type=str, choices=["colored", "shuffled"])
parser.add_argument('--model_type', type=str, choices=["linear", "resnet"])
parser.add_argument('--dataset', type=str, choices=["mnist", "cifar100"])
parser.add_argument('--eval_model_type', type=str, default="last")
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--ratio', type=float, default=0.15)

parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--save_models', action='store_true')
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--check_performance', action="store_true")
parser.add_argument('--get_entropy', action="store_true")
parser.add_argument('--get_mi', action="store_true")
parser.add_argument('--load_pretrained', action="store_true")

parser.add_argument('--early_stopping', action='store_true')
parser.add_argument('--min_epochs_before_stop', type=int, default=201)
parser.add_argument('--min_loss_before_stop', type=float, default=0.2)
parser.add_argument('--stopping_patience', type=int, default=3)

flags = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

def pretty_print(*values):
	col_width = 13
	def format_val(v):
		if not isinstance(v, str):
			v = np.array2string(v, precision=5, floatmode='fixed')
		return v.ljust(col_width)
	str_values = [format_val(v) for v in values]
	print("   ".join(str_values))

class CustomDataset(Dataset):
	def __init__(self, images, targets):
		self.images = images
		self.targets = targets
	
	def __getitem__(self, index):
		image = self.images[index]
		target = self.targets[index]
		
		return image, target
	
	def __len__(self):
		return len(self.images)

def train_resnet():
	seed = flags.seed
	print("Seed:", seed)
	pl.seed_everything(seed)
	set_seed(seed)
	envs =  prepare_data(flags.mode, seed, flags.ratio, dataset=flags.dataset)
	mnist_train = CustomDataset(envs[0]["images"], envs[0]["labels"])
	mnist_val = CustomDataset(envs[1]["images"], envs[1]["labels"])

	train_dl = DataLoader(mnist_train, batch_size=64)
	val_dl = DataLoader(mnist_val, batch_size=64)

	if flags.dataset == "mnist":
		model = ResNetMNIST(flags.mode)
	elif flags.dataset == "cifar100":
		model = ResNetCIFAR(flags.mode)
	
	# Init ModelCheckpoint callback, monitoring 'val_loss'
	checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="train_loss")
	train_stopping = pl.callbacks.EarlyStopping('train/loss', mode="min", stopping_threshold=0.01, patience=5)
	# val_stopping = pl.callbacks.EarlyStopping('val_loss', mode="min", patience=3)

	# logging using wandb
	project_name = "Colored-MNIST-ResNet" if flags.dataset == "mnist" else "Colored-CIFAR100-ResNet"
	wandb_logger = pl.loggers.WandbLogger(project=project_name, log_model="all",
										  name=f"{flags.mode}_ratio={flags.ratio:.3f}_seed={seed}")
	
	save_dir = f'models/{flags.mode}/dataset={flags.dataset}_ratio={flags.ratio:.3f}_l2-reg={flags.l2_regularizer_weight:.3f}_penalty-wt={flags.penalty_weight:.3f}/seed={seed}/resnet/'
	num_epochs = flags.epochs
	
	if os.path.exists(save_dir):
		checkpoint_name = [x for x in os.listdir(f'{save_dir}/{project_name}') if "checkpoints" in os.listdir(f'{save_dir}/{project_name}/{x}')][0]
		checkpoint_dir = f'{save_dir}/{project_name}/{checkpoint_name}/checkpoints/'
		checkpoint_path = checkpoint_dir + os.listdir(checkpoint_dir)[0]
		print(f'Loading from {checkpoint_path}')
		epoch = int(os.listdir(checkpoint_dir)[0].split('-')[0].split('=')[1])
		num_epochs = epoch + flags.epochs
		checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir)
	
	trainer = pl.Trainer(
		gpus=1, # use one GPU
		accelerator="auto",
		max_epochs=num_epochs,
		check_val_every_n_epoch=1,
		callbacks=[train_stopping] if not os.path.exists(save_dir) else [checkpoint_callback],
		enable_checkpointing=True,
		default_root_dir=save_dir,
		weights_save_path=save_dir,
		enable_progress_bar=True,
		logger=wandb_logger,
		resume_from_checkpoint=checkpoint_path if os.path.exists(save_dir) else None
	)
	# wandb_logger.watch(model)
	trainer.fit(model, train_dl, val_dl)

def train_linear():
	model_conf = '_'.join([str(flags.hidden_dim)]*flags.num_layers)
	seed = flags.seed
	print("Seed", seed)
	set_seed(seed)

	save_dir = f'models/{flags.mode}/ratio={flags.ratio:.3f}_l2-reg={flags.l2_regularizer_weight:.3f}_penalty-wt={flags.penalty_weight:.3f}/seed={seed}/model={model_conf}/'
	os.makedirs(save_dir, exist_ok=True)

	# get data environments
	envs = prepare_data(flags.mode, seed, flags.ratio)

	# instantiate the model
	set_seed(seed)
	if flags.load_pretrained:
		print("Loading pre-trained model")
		mlp = get_model(save_dir, flags.hidden_dim, flags.grayscale_model, flags.mode, flags.num_layers)
	else:
		mlp = MLP(flags.hidden_dim, flags.grayscale_model, flags.mode, flags.num_layers).to(device)

	# Train loop
	optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

	pretty_print('epoch', 'train nll', 'train acc', 'train penalty', 'val acc')
	
	best_val_acc, patience_tests = 0, 0
	train_size, val_size = len(envs[0]['images']), len(envs[1]['images'])
	print(f'Training set size: {train_size}, Validation set size: {val_size}')
	
	for epoch in range(flags.epochs):
		buffer_accs, buffer_losses = [], []
		for batch_idx in range(train_size//flags.batch_size):
			train_batch_imgs = envs[0]['images'][flags.batch_size*batch_idx : flags.batch_size*(batch_idx+1)]
			train_batch_labels = envs[0]['labels'][flags.batch_size*batch_idx : flags.batch_size*(batch_idx+1)]

			train_logits = mlp(train_batch_imgs)
			train_loss = mean_nll(flags.mode, train_logits, train_batch_labels).mean()
			train_acc = mean_accuracy(flags.mode, train_logits, train_batch_labels).mean()
			train_penalty = penalty(flags.mode, train_logits, train_batch_labels).mean()

			weight_norm = torch.tensor(0.).to(device)
			for w in mlp.parameters():
				weight_norm += w.norm().pow(2)
			
			loss = train_loss.clone()
			loss += flags.l2_regularizer_weight * weight_norm
			penalty_weight = (flags.penalty_weight
				if epoch >= flags.penalty_anneal_iters else 1.0)
			loss += penalty_weight * train_penalty
			if penalty_weight > 1.0:
				# Rescale the entire loss to keep gradients in a reasonable range
				loss /= penalty_weight

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			val_logits = mlp(envs[1]['images'])
			val_acc = mean_accuracy(flags.mode, val_logits, envs[1]['labels']).mean()

			buffer_accs.append(val_acc.detach().cpu().numpy())
			buffer_losses.append(train_loss.detach().cpu().numpy())

		if epoch % 10 == 0:
			pretty_print(
			np.int32(epoch),
			train_loss.detach().cpu().numpy(),
			train_acc.detach().cpu().numpy(),
			train_penalty.detach().cpu().numpy()*penalty_weight,
			val_acc.detach().cpu().numpy()
			)

			if flags.save_models:
				torch.save(mlp.state_dict(), save_dir + f"epoch={epoch}.pt")
			
			if flags.early_stopping:
				if best_val_acc < np.mean(buffer_accs):
					best_val_acc = val_acc
					torch.save(mlp.state_dict(), save_dir + f"best.pt")
					patience_tests = 0
				else:
					# scheduler.step()
					if epoch >= flags.min_epochs_before_stop and np.mean(buffer_losses) <= flags.min_loss_before_stop:
						patience_tests += 1
						if patience_tests > flags.stopping_patience:
							print('EARLY STOPPING')
							break

def eval_linear():
	seed = flags.seed
	set_seed(seed)
	model_conf = '_'.join([str(flags.hidden_dim)]*flags.num_layers)
	dir = f'models/{flags.mode}/dataset={flags.dataset}_ratio={flags.ratio:.3f}_l2-reg={flags.l2_regularizer_weight:.3f}_penalty-wt={flags.penalty_weight:.3f}/seed={seed}/model={model_conf}/'

	model = get_model(dir, flags.hidden_dim, flags.grayscale_model, flags.mode, flags.num_layers, flags.eval_model_type)
	envs = prepare_data(flags.mode, seed, flags.ratio, dataset=flags.dataset)
	train_env, val_env = envs[0], envs[1]

	dir += flags.eval_model_type + "_" if flags.eval_model_type != "last" else ""
	if flags.check_performance:
		# val_logits = model(val_env['images'])
		# val_acc = mean_accuracy(val_logits, val_env['labels'])
		# print("Validation Accuracy:", val_acc)
		get_performance(train_env, val_env, model, dir, flags.mode)

	reps = get_reps(flags, model, val_env["images"], dir)

	if flags.get_entropy:
		entropy = get_entropy(flags, reps, dir, num_bins=100)
	if flags.get_mi:
		sq_mis = get_mi(flags, reps, dir)

	print("Done!")

def eval_resnet():
	seed = flags.seed
	set_seed(seed)
	model_conf = '_'.join([str(flags.hidden_dim)]*flags.num_layers)
	base_dir = f'models/{flags.mode}/dataset={flags.dataset}_ratio={flags.ratio:.3f}_l2-reg={flags.l2_regularizer_weight:.3f}_penalty-wt={flags.penalty_weight:.3f}/seed={seed}/resnet/'
	ext = os.listdir(base_dir)[0]
	dir = base_dir + f"{ext}/"
	ext = os.listdir(dir)[0]
	dir += f"{ext}/"
	ckpt = os.listdir(dir + "checkpoints")[0]

	if flags.dataset == "mnist":
		model = ResNetMNIST(flags.mode)
	elif flags.dataset == "cifar100":
		model = ResNetCIFAR(flags.mode)
	
	model = model.load_from_checkpoint(dir + f"checkpoints/{ckpt}", mode=flags.mode)
	model.eval()
	trainer = pl.Trainer(
        gpus=1, # use one GPU
        accelerator="auto")
	
	print(f'Loaded from {dir + f"checkpoints/{ckpt}"}')
	
	envs = prepare_data(flags.mode, seed, flags.ratio, dataset=flags.dataset)

	data_train = CustomDataset(envs[0]["images"], envs[0]["labels"])
	data_val = CustomDataset(envs[1]["images"], envs[1]["labels"])

	train_dl = DataLoader(data_train, batch_size=64)
	val_dl = DataLoader(data_val, batch_size=64)
	
	dir += flags.eval_model_type + "_" if flags.eval_model_type != "last" else ""
	if flags.check_performance:
		# def get_performance(model, dataloader):
		# 	model.eval()
		# 	model = model.to(device)
		# 	checks = []
		# 	batches = dataloader
		# 	for batch in batches:
		# 		x, y_true = batch
		# 		x = x.to(device)
		# 		with torch.no_grad():
		# 			logits = model(x)
		# 		checks.extend(list((y_true == torch.argmax(logits, 1)).view(-1)))
			
		# 	accuracy = sum(checks)/len(checks)
		# 	return accuracy
		
		# train_acc, val_acc = get_performance(model.model, train_dl), get_performance(model.model, val_dl)
		# print(f"Train accuracy: {train_acc:.2f}")
		# print(f"Validation accuracy: {val_acc:.2f}")

		train_acc = trainer.validate(model=model, dataloaders=[train_dl], verbose=False)
		val_acc = trainer.validate(model=model, dataloaders=[val_dl], verbose=False)
		print(f"Train: {train_acc}")
		print(f"Validation: {val_acc}")
	
	reps = get_reps(flags, model.model, val_dl, base_dir)
	
	if flags.get_entropy:
		entropy = get_entropy(flags, reps, base_dir, num_bins=100, global_binning=True)
	if flags.get_mi:
		sq_mis = get_mi(flags, reps, base_dir)
	
	print("Done!")

def main():
	if not flags.eval_only:
		if flags.model_type == "linear":
			train_linear()
		elif flags.model_type == "resnet":
			train_resnet()
	if flags.model_type == "linear":
		eval_linear()
	elif flags.model_type == "resnet":
		eval_resnet()

if __name__ == "__main__":
	main()