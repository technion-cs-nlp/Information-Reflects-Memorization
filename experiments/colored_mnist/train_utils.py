import torch
from torch import nn, autograd
import numpy as np
from torchvision.models import resnet18
import pytorch_lightning as pl
# from pytorch_lightning.core.decorators import auto_move_data
from torchvision import datasets

import sys
sys.path.append('../../utils')
from common_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss function helpers
def mean_nll(mode, logits, y):
  return nn.functional.binary_cross_entropy_with_logits(logits, y) if mode == "colored" \
    else nn.functional.cross_entropy(logits, y)

def mean_accuracy(mode, logits, y):
  if mode == "colored":
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()
  else:
    return ((torch.argmax(logits, dim=-1).float() - y) == 0).float().mean()

def penalty(mode, logits, y):
  scale = torch.tensor(1.).to(device).requires_grad_()
  loss = mean_nll(mode, logits * scale, y)
  grad = autograd.grad(loss, [scale], create_graph=True)[0]
  return torch.sum(grad**2)

def make_colored_environment(images, labels, e, dataset):
    def torch_bernoulli(p, size):
      return (torch.rand(size) < p).float()
    def torch_xor(a, b):
      return (a-b).abs() # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.).to(device),
      'labels': labels[:, None].to(device)
    }

def make_shuffled_environment(images, labels, e, dataset):
	if dataset == "mnist":
		# 2x subsample for computational convenience
		images = images.reshape((-1, 28, 28))[:, ::2, ::2]
		images = torch.stack([images, images], dim=1)
		num_labels = 10
	elif dataset == "cifar100":
		images = torch.from_numpy(images.reshape((-1, 3, 32, 32)))
		num_labels = 100
	else:
		raise ValueError()
	
	# shuffle labels
	to_assign = np.random.rand(len(labels))
	bin_ranges = np.linspace(0, 1*e, num=num_labels+1)
	labels_ = np.digitize(to_assign, bins=bin_ranges)-1
	labels_[labels_ == num_labels] = np.array(labels)[labels_ == num_labels]
	print(f'Perc flips: {np.sum(labels_ != np.array(labels)).item()/len(labels)}')
	labels = torch.tensor(labels_, dtype=torch.long).view(-1)
	print(labels.shape)
	assert len(np.unique(labels)) == num_labels

	return {
	'images': (images.float() / 255.).to(device),
	'labels': labels.to(device)
	}

def prepare_data(mode, seed, ratio, dataset="mnist"):
	set_seed(seed)
	
	if dataset == "mnist":
		# Load MNIST, make train/val splits, and shuffle train set examples
		data = datasets.MNIST('~/datasets/mnist', train=True, download=True)
		data_train = (data.data[:50000], data.targets[:50000])
		data_val = (data.data[50000:], data.targets[50000:])
	elif dataset == "cifar100":
		# Load CIFAR100, make train/val splits, and shuffle train set examples
		data = datasets.CIFAR100('~/datasets/cifar100_train', train=True, download=True)
		data_train = (data.data[:], data.targets[:])
		print("Number of images in CIFAR:", len(data_train[0]), "size of each:", data_train[0][0].shape)
		data = datasets.CIFAR100('~/datasets/cifar100_test', train=False, download=True)
		data_val = (data.data[:], data.targets[:])
	else:
		raise ValueError()
	
	rng_state = np.random.get_state()
	np.random.shuffle(data_train[0].numpy() if dataset == "mnist" else data_train[0])
	np.random.set_state(rng_state)
	np.random.shuffle(data_train[1].numpy() if dataset == "mnist" else data_train[1])

    # Build environments
	make_environment = make_colored_environment if mode == "colored" else make_shuffled_environment
	envs = [
        make_environment(data_train[0], data_train[1], ratio, dataset),
        make_environment(data_val[0], data_val[1], 0.5 if mode == "colored" else 0.0, dataset)
    ]
	return envs

class MLP(nn.Module):
	def __init__(self, hidden_dim, grayscale_model: bool, mode: str, num_layers:int=2):
		super(MLP, self).__init__()
		self.hidden_dim = hidden_dim
		self.grayscale_model = grayscale_model
		self.mode = mode
		self.num_layers = num_layers
		self.lins = []
		if self.grayscale_model or self.mode == "shuffled":
			self.lin1 = nn.Linear(14 * 14, self.hidden_dim)
		else:
			self.lin1 = nn.Linear(2 * 14 * 14, self.hidden_dim)
		self.lins.append(self.lin1)
		self.lins.append(nn.ReLU(True))
		if self.num_layers >= 2:
			self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
			self.lins.append(self.lin2)
			self.lins.append(nn.ReLU(True))
		if self.num_layers >= 3:
			self.lin3 = nn.Linear(self.hidden_dim, self.hidden_dim)
			self.lins.append(self.lin3)
			self.lins.append(nn.ReLU(True))
		if self.num_layers >= 4:
			self.lin4 = nn.Linear(self.hidden_dim, self.hidden_dim)
			self.lins.append(self.lin4)
			self.lins.append(nn.ReLU(True))
		self.linC = nn.Linear(self.hidden_dim, 1) if self.mode == "colored" else nn.Linear(self.hidden_dim, 10)
		for lin in self.lins[::2]:
			nn.init.xavier_uniform_(lin.weight)
			nn.init.zeros_(lin.bias)
		self._main = nn.Sequential(*self.lins, self.linC)

	def forward(self, input):
		if self.grayscale_model or self.mode == "shuffled":
			out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
		else:
			out = input.view(input.shape[0], 2 * 14 * 14)
			out = self._main(out)
		return out

class ResNetMNIST(pl.LightningModule):
	def __init__(self, mode):
		super().__init__()
		# define model and loss
		num_classes = 1 if mode=="colored" else 10
		self.model = resnet18(num_classes=num_classes)
		self.model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		self.loss = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
	
	def accuracy(self, output, y):
		return torch.sum(torch.argmax(output, 1) == y)/len(output)

	# @auto_move_data # this decorator automatically handles moving your tensors to GPU if required
	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_no):
		# implement single training step
		x, y = batch
		logits = self(x)
		loss = self.loss(logits, y)
		self.log("train/loss", loss)
		return loss

	def configure_optimizers(self):
		# choose your optimizer
		return torch.optim.RMSprop(self.parameters(), lr=0.005)

	def validation_step(self, batch, batch_nb):
		x, y = batch
		logits = self.forward(x)
		loss = self.loss(logits, y)
		acc = self.accuracy(logits, y)

		self.log("val_loss", loss, on_epoch=True)
		self.log("val_acc", acc, on_epoch=True)

		return {'batch_val_loss': loss,
				'batch_val_acc': acc}

class ResNetCIFAR(pl.LightningModule):
	def __init__(self, mode):
		super().__init__()
		# define model and loss
		num_classes = 1 if mode=="colored" else 100
		self.model = resnet18(num_classes=num_classes)
		self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
		self.loss = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
	
	def accuracy(self, output, y):
		return torch.sum(torch.argmax(output, 1) == y)/len(output)

	# @auto_move_data # this decorator automatically handles moving your tensors to GPU if required
	def forward(self, x):
		return self.model(x)

	def training_step(self, batch, batch_no):
		# implement single training step
		x, y = batch
		logits = self(x)
		loss = self.loss(logits, y)
		self.log("train/loss", loss)
		return loss

	def configure_optimizers(self):
		# choose your optimizer
		return torch.optim.RMSprop(self.parameters(), lr=0.005)

	def validation_step(self, batch, batch_nb):
		x, y = batch
		logits = self.forward(x)
		loss = self.loss(logits, y)
		acc = self.accuracy(logits, y)

		self.log("val_loss", loss, on_epoch=True)
		self.log("val_acc", acc, on_epoch=True)

		return {'batch_val_loss': loss,
				'batch_val_acc': acc}