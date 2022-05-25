import argparse
import random
from cProfile import label
from multiprocessing.spawn import prepare
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision.models import resnet18
import PIL
import wandb

from eval_utils import get_reps, get_entropy, get_mi, check_performance

import sys
sys.path.append('../../utils')
from common_utils import set_seed

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

parser = argparse.ArgumentParser(description='NICO')
parser.add_argument('--eval_model_type', type=str, default="last")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--training_set', type=str, choices=["1", "2", "1_plus", "2_plus"])

parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--check_performance', action="store_true")
parser.add_argument('--get_entropy', action="store_true")
parser.add_argument('--get_mi', action="store_true")
parser.add_argument('--load_pretrained', action="store_true")

parser.add_argument('--num_neurons', type=int, default=768,
                    help="number of neurons for which to compute MI")
parser.add_argument('--num_bins', type=int, default=100,
                    help="number of bins to discretize entropy")
parser.add_argument('--num_neighbors', type=int, default=3,
                    help="number of k-nearest neighbours to compute MI")

flags = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Flags:')
for k,v in sorted(vars(flags).items()):
  print("\t{}: {}".format(k, v))

animal_classes = {
    "bear" : ["in forest", "in water", "eating grass", "on snow", "on ground"],
    "dog" : ["on beach", "on snow", "on grass", "lying", "in water"],
    "cat" : ["at home", "in street", "in river", "on snow", "on grass"],
    "bird" : ["flying", "on ground", "on grass", "on branch", "in hand"],
    "horse" : ["on grass", "on beach", "in forest", "on snow", "in river"]
}

animal_classes_plus = {
    "bear" : "grass",
    "dog" : "water",
    "cat" : "autumn",
    "bird" : "dim",
    "horse" : "outdoor",
    "sheep" : "rock"
}

label_map = {
    class_name : class_idx for class_idx, class_name in enumerate(animal_classes)
}

label_map_plus = {
    class_name : class_idx for class_idx, class_name in enumerate(animal_classes_plus)
}

# train_set = [
#     [227, 155, 114, 94, 79],
#     [267, 235, 143, 124, 123],
#     [268, 160, 137, 125, 123],
#     [191, 260, 228, 226, 77],
#     [147, 147, 126, 120, 54]
# ]

train_set_1 = [
    [227, 75, 71, 50, 50],
    [267, 67, 66, 50, 50],
    [268, 67, 67, 50, 50],
    [191, 90, 78, 60, 58],
    [147, 100, 80, 73, 54]
]

train_set_2 = [
    [100, 100, 100, 94, 79],
    [100, 100, 100, 100, 100],
    [100, 100, 100, 100, 100],
    [100, 100, 100, 100, 77],
    [100, 100, 100, 100, 54]
]

val_set = [
    [20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20],
    [20, 20, 20, 20, 20]
]

wandb_key = open("../../wandb.config", "r").read()
wandb.login(key=wandb_key)

def prepare_data():
    train_images, train_labels = [], []
    val_images, val_labels = [], []

    def read_image(path):
        # path = path.replace(' ', '\ ')
        img =  PIL.Image.open(path)
        img = np.asarray(img)
        # print(img.shape)
        img = np.resize(img, (64, 64, 3))
        img = torch.tensor(img).float() / 255.
        img = img.moveaxis(2, 0)

        return img
    
    if "plus" in flags.training_set:
        if '1' in flags.training_set:
            uniform = False
        elif '2' in flags.training_set:
            uniform = True
        
        for idx_1, animal in enumerate(animal_classes_plus):
            base_dir = 'dataset_plus/without_context/train'
            for img_ in os.listdir(f'{base_dir}/{animal}'):
                try:
                    img = read_image(f'{base_dir}/{animal}/{img_}')
                except:
                    print(f"Warning: Can't read image {base_dir}/{animal}/{img_}")
                    continue
                train_images.append(img)
                train_labels.append(label_map_plus[animal])

            base_dir = 'dataset_plus/with_context/train'
            major_dir = f'{base_dir}/{animal_classes_plus[animal]}/{animal}'
            num_imgs = len(os.listdir(major_dir))
            num_imgs = num_imgs // len(animal_classes_plus)
            rem = num_imgs % len(animal_classes_plus)

            for idx_2, context in enumerate(os.listdir(base_dir)):
                imgs = os.listdir(f'{base_dir}/{context}/{animal}')
                if context != animal_classes_plus[animal]:
                    train_imgs = np.random.choice(imgs, num_imgs)
                    val_imgs = list(set(imgs) ^ set(train_imgs))
                    for img_ in val_imgs:
                        try:
                            img = read_image(f'{base_dir}/{context}/{animal}/{img_}')
                        except:
                            print(f"Warning: Can't read image {base_dir}/{context}/{animal}/{img_}")
                            continue
                        val_images.append(img)
                        val_labels.append(label_map_plus[animal])
                
                if uniform:
                    train_imgs = np.random.choice(imgs, num_imgs)
                elif context == animal_classes_plus[animal]:
                    train_imgs = imgs[:len(imgs)-rem]
                else:
                    continue
                
                for img_ in train_imgs:
                    try:
                        img = read_image(f'{base_dir}/{context}/{animal}/{img_}')
                    except:
                        print(f"Warning: Can't read image {base_dir}/{context}/{animal}/{img_}")
                        continue
                    train_images.append(img)
                    train_labels.append(label_map_plus[animal])

    else:
        if flags.training_set == '1':
            train_set = train_set_1
        elif flags.training_set == '2':
            train_set = train_set_2
        else:
            raise ValueError()
        
        for idx_1, dir_1 in enumerate(animal_classes):
            for idx_2, dir_2 in enumerate(animal_classes[dir_1]):
                sclass_images = []
                for img_ in os.listdir(f'dataset/Animal/{dir_1}/{dir_2}'):
                    try:
                        img = read_image(f'dataset/Animal/{dir_1}/{dir_2}/{img_}')
                    except:
                        print(f"Warning: Can't read image dataset/Animal/{dir_1}/{dir_2}/{img_}")
                        continue
                    # img = torch.tensor(img).float() / 255.
                    # img = img.moveaxis(2, 0)
                    sclass_images.append(img)
                indices = list(range(len(sclass_images)))
                train_ids = np.random.choice(indices, train_set[idx_1][idx_2])
                val_choices = set(indices) ^ set(train_ids)
                val_ids = np.random.choice(list(val_choices), val_set[idx_1][idx_2])
                for idx in train_ids:
                    train_images.append(sclass_images[idx])
                    train_labels.append(label_map[dir_1])
                for idx in val_ids:
                    val_images.append(sclass_images[idx])
                    val_labels.append(label_map[dir_1])
    
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)
    
    return [
        {'images': train_images, 'labels': train_labels.to(device)},
        {'images': val_images, 'labels': val_labels.to(device)}
    ]

class CustomNICO(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets
        
    def __getitem__(self, index):
        image = self.images[index].to(device)
        target = self.targets[index]
        
        return image, target

    def __len__(self):
        return len(self.images)

def get_dataloaders(envs, seed):
    print(f'Training size: {len(envs[0]["images"])}')
    print(f'Validation size: {len(envs[1]["images"])}')

    nico_train = CustomNICO(envs[0]["images"], envs[0]["labels"])
    nico_val = CustomNICO(envs[1]["images"], envs[1]["labels"])

    def seed_worker(worker_id):
        np.random.seed(seed)
        random.seed(seed)
    
    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = DataLoader(nico_train, batch_size=flags.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_dl = DataLoader(nico_val, batch_size=flags.batch_size)

    return train_dl, val_dl

class ResNetNICO(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # define model and loss
        num_classes = 6 if "plus" in flags.training_set else 5
        self.model = resnet18(num_classes=num_classes, pretrained=False)
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
        acc = self.accuracy(logits, y)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss

    def configure_optimizers(self):
        # choose your optimizer
        return torch.optim.RMSprop(self.parameters(), lr=1e-5)
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        acc = self.accuracy(logits, y)

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

        return {'batch_val_loss': loss,
                'batch_val_acc': acc}

def train():
    seed = flags.seed
    print("Seed", seed)
    pl.seed_everything(seed)
    set_seed(seed)

    save_dir = f'models/set_{flags.training_set}/seed={seed}/resnet/'

    envs =  prepare_data()
    train_dl, val_dl = get_dataloaders(envs, seed)

    model = ResNetNICO()
    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=save_dir, save_top_k=3, monitor="val_loss")
    train_stopping = pl.callbacks.EarlyStopping('train/loss', mode="min", stopping_threshold=0.1)
    # val_stopping = pl.callbacks.EarlyStopping('val_loss', mode="min", patience=5)

    # logging using wandb
    project_name_ext = "-Plus" if "plus" in flags.training_set else ""
    wandb_logger = pl.loggers.WandbLogger(project="NICO-ResNet" + project_name_ext, log_model="all", name=f"set={flags.training_set}_seed={seed}")

    trainer = pl.Trainer(
        gpus=1, # use one GPU
        accelerator="auto",
        max_epochs=flags.epochs,
        check_val_every_n_epoch=1,
        # callbacks=[train_stopping, checkpoint_callback],
        callbacks=[checkpoint_callback],
        # enable_checkpointing=True,
        default_root_dir=save_dir,
        # weights_save_path=save_dir,
        enable_progress_bar=True,
        logger=wandb_logger
    )
    # wandb_logger.watch(model)
    trainer.fit(model, train_dl, val_dl)
    # trainer.validate(ckpt_path="best")

def eval():
    seed = flags.seed
    set_seed(seed)
    pl.seed_everything(seed)
    envs =  prepare_data()
    train_dl, val_dl = get_dataloaders(envs, seed)

    dir = f'models/set_{flags.training_set}/seed={seed}/resnet/'
    ckpt = sorted_alphanumeric([x for x in os.listdir(dir) if "ckpt" in x])[-1]
    print(f'Loading checkpoint: {ckpt}')

    # model = ResNetNICO()
    # checkpoint_data = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint_data['state_dict'])
    
    model = ResNetNICO.load_from_checkpoint(dir + ckpt)
    model.eval()

    trainer = pl.Trainer(
        gpus=1, # use one GPU
        accelerator="auto")
    
    # dir += flags.eval_model_type + "_" if flags.eval_model_type != "last" else ""
    if flags.check_performance:
        # val_logits = model(val_env['images'])
        # val_acc = mean_accuracy(val_logits, val_env['labels'])
        # print("Validation Accuracy:", val_acc)
        # train_acc, val_acc = check_performance(model.model, train_dl), check_performance(model.model, val_dl)
        train_acc = trainer.validate(model=model, dataloaders=[train_dl], verbose=False)
        val_acc = trainer.validate(model=model, dataloaders=[val_dl], verbose=False)
        print(f"Train: {train_acc}")
        print(f"Validation: {val_acc}")

        with open(dir + 'performance.txt', 'w') as f:
            f.write(f"Train Acc: {train_acc[0]['val_acc']}\tTrain Loss: {train_acc[0]['val_loss']}\n")
            f.write(f"Val Acc: {val_acc[0]['val_acc']}\tVal Loss: {val_acc[0]['val_loss']}")
    
    reps = get_reps(model.model, val_dl, dir)
    
    if flags.get_entropy:
        entropy = get_entropy(reps, dir, num_bins=flags.num_bins, global_binning=True)
    if flags.get_mi:
        sq_mis = get_mi(flags, reps, dir)
    
    print("Done!")

if __name__ == "__main__":
    if not flags.eval_only:
        train()
    eval()