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
class Sequential(nn.Module):
    _modules: Dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        if index < 0: index += len(self._modules) # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        if index < 0: index += len(self._modules) # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x


class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.weight = t.nn.Parameter(t.ones((num_features,)))
        self.bias = t.nn.Parameter(t.zeros((num_features,)))
        self.register_buffer('running_mean', t.zeros((num_features,)))
        self.register_buffer('running_var', t.ones((num_features,)))
        self.register_buffer('num_batches_tracked', t.tensor(0))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        if self.training:
            v = t.var(x, dim=(0,2,3), unbiased=False, keepdim=True)
            m = t.mean(x, dim=(0,2,3), keepdim=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * m.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * v.squeeze()
            self.num_batches_tracked += 1
        else:
            m = self.running_mean.reshape(1,-1,1,1)
            v = self.running_var.reshape(1,-1,1,1)

        x = (x - m) / t.sqrt(v + self.eps) * self.weight.reshape(1,-1,1,1) + self.bias.reshape(1,-1,1,1)
        return x

    def extra_repr(self) -> str:
        pass


if MAIN:
    tests.test_batchnorm2d_module(BatchNorm2d)
    tests.test_batchnorm2d_forward(BatchNorm2d)
    tests.test_batchnorm2d_running_mean(BatchNorm2d)
# %%
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return t.mean(x, dim=(2, 3))


class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.left = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(out_feats)
        )
        if first_stride > 1:
            self.right = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride),
                BatchNorm2d(out_feats)
            )
        else:
            self.right = nn.Identity()

        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        return self.relu(self.left(x) + self.right(x))

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        bs = [ResidualBlock(in_feats, out_feats, first_stride)] 
        bs += [ResidualBlock(out_feats, out_feats) for n in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*bs)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.blocks(x)


class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        input_feat_num = 64
        self.in_layers = Sequential(
            Conv2d(3, input_feat_num, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(input_feat_num),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        all_in_feats = [input_feat_num] + out_features_per_group[:-1]
        block_groups = [
            BlockGroup(*args) for args 
            in zip(
                n_blocks_per_group,
                all_in_feats,
                out_features_per_group,
                first_strides_per_group,
            )
        ]
        self.blocks = Sequential(*block_groups)

        self.out_layers = Sequential(
            AveragePool(),
            Flatten(),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        x = self.in_layers(x)
        x = self.blocks(x)
        x = self.out_layers(x)
        return x


if MAIN:
    my_resnet = ResNet34()

# %%
def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    # Get the state dictionaries for each model, check they have the same number of parameters & buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet



if MAIN:
    pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    my_resnet = copy_weights(my_resnet, pretrained_resnet)
# %%
if MAIN:
    print_param_count(my_resnet, pretrained_resnet)
# %%
if MAIN:
    IMAGE_FILENAMES = [
        "chimpanzee.jpg",
        "golden_retriever.jpg",
        "platypus.jpg",
        "frogs.jpg",
        "fireworks.jpg",
        "astronaut.jpg",
        "iguana.jpg",
        "volcano.jpg",
        "goofy.jpg",
        "dragonfly.jpg",
    ]

    IMAGE_FOLDER = section_dir / "resnet_inputs"

    images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]
# %%

if MAIN:
    display(images[4])
# %%
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# %%
def prepare_data(images: List[Image.Image]) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    return t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0)


if MAIN:
    prepared_images = prepare_data(images)
    assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)
# %%
def predict(model, images):
    logits: t.Tensor = model(images)
    return logits.argmax(dim=1)

if MAIN:
    with open(section_dir / "imagenet_labels.json") as f:
        imagenet_labels = list(json.load(f).values())
# %%
# Check your predictions match the pretrained model's

if MAIN:
    my_predictions = predict(my_resnet, prepared_images)
    pretrained_predictions = predict(pretrained_resnet, prepared_images)
    assert all(my_predictions == pretrained_predictions)
    print('match')
# %%
# Print out your predictions, next to the corresponding images

if MAIN:
    for img, label in zip(images, my_predictions):
        print(f"Class {label}: {imagenet_labels[label]}")
        display(img)
        print()
# %%

class NanModule(nn.Module):
    '''
    Define a module that always returns NaNs (we will use hooks to identify this error).
    '''
    def forward(self, x):
        return t.full_like(x, float('nan'))


if MAIN:
    model = nn.Sequential(
        nn.Identity(),
        NanModule(),
        nn.Identity()
    )


def hook_check_for_nan_output(module: nn.Module, input: Tuple[t.Tensor], output: t.Tensor) -> None:
    '''
    Hook function which detects when the output of a layer is NaN.
    '''
    if t.isnan(output).any():
        print('Check')
        raise ValueError(f"NaN output from {module}")


def add_hook(module: nn.Module) -> None:
    '''
    Register our hook function in a module.

    Use model.apply(add_hook) to recursively apply the hook to model and all submodules.
    '''
    module.register_forward_hook(hook_check_for_nan_output)


def remove_hooks(module: nn.Module) -> None:
    '''
    Remove all hooks from module.

    Use module.apply(remove_hooks) to do this recursively.
    '''
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()



if MAIN:
    model = model.apply(add_hook)
    input = t.randn(3)

    try:
        output = model(input)
    except ValueError as e:
        print(e)

    model = model.apply(remove_hooks)
# %%
