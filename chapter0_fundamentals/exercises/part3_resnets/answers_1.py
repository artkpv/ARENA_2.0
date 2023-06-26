#%%
import os

from pytorch_lightning.utilities.types import STEP_OUTPUT; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange
from dataclasses import dataclass
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from typing import Any, List, Optional, Tuple, Dict
from PIL import Image
from IPython.display import display
from pathlib import Path
import torchinfo
import json
import pandas as pd
from jaxtyping import Float, Int
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_resnets"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from part2_cnns.solutions import get_mnist, Linear, Conv2d, Flatten, ReLU, MaxPool2d
from part3_resnets.utils import print_param_count
import part3_resnets.tests as tests
from plotly_utils import line, plot_train_loss_and_test_accuracy_from_metrics

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

MAIN = __name__ == "__main__"

pl.seed_everything()


# %%

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequencial = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1,padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

			nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3,padding=1,stride=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

			nn.Flatten(),
			nn.Linear(in_features=3136, out_features=128),
			nn.Linear(in_features=128, out_features=10)
		)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.sequencial(x)


if MAIN:
    model = ConvNet()
    print(model)

if MAIN:
    summary = torchinfo.summary(model, input_size=(1, 1, 28, 28))
    print(summary)
# %%
if MAIN:
    MNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

if MAIN:
    data_augmentation_transform = transforms.Compose([
       transforms.RandomRotation(degrees=15),
       transforms.RandomResizedCrop(size=28, scale=(0.8, 1.2)),
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomVerticalFlip(p=0.5),
       transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_mnist(subset: int = 1, train_transform=None, test_transform=None):
    if train_transform is None:
        train_transform = MNIST_TRANSFORM
    if test_transform is None:
        test_transform = MNIST_TRANSFORM

    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    ts_l = len(mnist_trainset)
    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, ts_l, subset))
        ts_l = len(mnist_trainset)
        val_l = ts_l // 10
        mnist_valset = Subset(mnist_trainset, indices=range(ts_l - val_l, ts_l))
        mnist_trainset = Subset(mnist_trainset, indices=range(0, ts_l - val_l))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))
    else:
        val_l = ts_l // 10
        mnist_trainset = Subset(mnist_trainset, indices=range(0, ts_l - val_l))
        mnist_valset = Subset(mnist_trainset, indices=range(ts_l - val_l, ts_l))

    return mnist_trainset, mnist_testset, mnist_valset


#%%
from tqdm.notebook import tqdm
import time


if MAIN:
    for i in tqdm(range(100)):
        time.sleep(0.01)

# %%
if MAIN:
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)


# %%
@dataclass
class ConvNetTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as 
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = ConvNetTrainingArgs(batch_size=128).
    '''
    batch_size: int = 64
    max_epochs: int = 15
    optimizer: t.optim.Optimizer = t.optim.Adam
    learning_rate: float = 1e-3
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day4-convenet"
    log_every_n_steps: int = 1
    sample: int = 10

    def __post_init__(self):
        '''
        This code runs after the class is instantiated. It can reference things like
        self.sample, which are defined in the __init__ block.
        '''
        trainset, testset, valset = get_mnist(subset=self.sample, train_transform=data_augmentation_transform, test_transform=MNIST_TRANSFORM)
        self.trainloader = DataLoader(trainset, shuffle=True, batch_size=self.batch_size)
        self.valloader = DataLoader(valset, shuffle=True, batch_size=self.batch_size)
        self.testloader = DataLoader(testset, shuffle=False, batch_size=self.batch_size)
        self.logger = CSVLogger(save_dir=self.log_dir, name=self.log_name)


class LitConvNet(pl.LightningModule):
    def __init__(self, args: ConvNetTrainingArgs):
        super().__init__()
        self.convnet = ConvNet()
        self.args = args

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int) -> t.Tensor:
        imgs, labels = batch
        logits = self.convnet(imgs)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = self.args.optimizer(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], batch_idx: int):
        imgs, labels = batch
        logits = self.convnet(imgs)

        # calculate acc
        labels_hat = t.argmax(logits, dim=1)
        val_acc = t.sum(labels == labels_hat).item() / (len(labels) * 1.0)

        # log the outputs!
        self.log('accuracy', val_acc)

#%%
if MAIN:
    args = ConvNetTrainingArgs()
    model = LitConvNet(args)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=args.logger,
        log_every_n_steps=1
    )
    trainer.validate(model=model, dataloaders=args.valloader)
    trainer.fit(model=model, train_dataloaders=args.trainloader, val_dataloaders=args.valloader)
# %%
if MAIN:
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    plot_train_loss_and_test_accuracy_from_metrics(metrics, "Training ConvNet on MNIST data")

# %%
