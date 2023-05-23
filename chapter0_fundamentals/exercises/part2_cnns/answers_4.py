#%%
import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

from collections import namedtuple

from part2_cnns.answers_3 import maxpool2d, conv2d


MAIN = __name__ == "__main__"

#%%
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding
        )

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return f'{self.kernel_size} KS, {self.stride} S, {self.padding} P'


if MAIN:
    tests.test_maxpool2d_module(MaxPool2d)
    m = MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")


# %%

class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.maximum(x, t.tensor(0))


if MAIN:
    tests.test_relu(ReLU)

# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim


    def forward(self, input: t.Tensor) -> t.Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        def num2char(n):
            res = ['a']
            while n > 0:
                n, r = divmod(n, 10)
                res += [chr(ord('a') + r)]
            return ''.join(res)

        dims = len(input.shape)
        r = dims + self.end_dim if self.end_dim < 0 else self.end_dim
        l = self.start_dim
        if not l < r:
             return input
        pattern = [num2char(i) for i in range(dims)]
        out_p = pattern[:]
        out_p[l] = '(' + out_p[l]
        out_p[r] = out_p[r] + ')'
        opspattern = ' '.join(pattern) + ' -> ' + ' '.join(out_p)
        return einops.rearrange(input, opspattern)

    def extra_repr(self) -> str:
        return f'{self.start_dim}-{self.end_dim}'


if MAIN:
    tests.test_flatten(Flatten)

# %%

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()

        # t.nn.init.xavier_uniform_(self.weight)
        sqrti = 1 / np.sqrt(in_features)
        self.weight = nn.Parameter(sqrti * (2 * t.rand(out_features, in_features) - 1))
        self.bias = nn.Parameter(sqrti * (2*t.rand(out_features,) - 1)) if bias else None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        res = x @ self.weight.T 
        if self.bias is not None:
            res += self.bias
        return res

    def extra_repr(self) -> str:
        return f'{self.weight.shape=} {self.bias.shape if self.bias is not None else ""}'



if MAIN:
    tests.test_linear_forward(Linear)
    tests.test_linear_parameters(Linear)
    tests.test_linear_no_bias(Linear)
# %%
class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        kh, kw = force_pair(kernel_size)
        sqrti = 1 / np.sqrt(in_channels * kh * kw)
        self.weight = nn.Parameter(sqrti * (2*t.rand(out_channels, in_channels, kh, kw) - 1))
        #t.nn.init.xavier_uniform_(self.weight)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return conv2d(x, self.weight, self.stride, self.padding)


    def extra_repr(self) -> str:
        return f'{self.weight.shape=} {self.stride=} {self.padding=}'



if MAIN:
    tests.test_conv2d_module(Conv2d)

# %%
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = ReLU()
        self.flatten = Flatten()
        self.fc = Linear(in_features=32*14*14, out_features=10)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.fc(self.flatten(self.relu(self.maxpool(self.conv(x)))))



if MAIN:
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    print(model)

# %%
if MAIN:
    MNIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset



if MAIN:
    mnist_trainset, mnist_testset = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=True)
# %%

if MAIN:
    img, label = mnist_trainset[1]

    imshow(
        img.squeeze(), 
        color_continuous_scale="gray", 
        zmin=img.min().item(),
        zmax=img.max().item(),
        title=f"Digit = {label}",
        width=450,
    )
# %%
if MAIN:
    img_input = img.unsqueeze(0).to(device) # add batch dimension
    probs = model(img_input).squeeze().softmax(-1).detach()

    bar(
        probs,
        x=range(1, 11),
        template="ggplot2",
        width=600,
        title="Classification probabilities", 
        labels={"x": "Digit", "y": "Probability"}, 
        text_auto='.2f',
        showlegend=False, 
        xaxis_tickmode="linear"
    )
# %%
if MAIN:
    batch_size = 64
    epochs = 3

    mnist_trainset, _ = get_mnist(subset = 10)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)

    optimizer = t.optim.Adam(model.parameters())
    loss_list = []

    for epoch in tqdm(range(epochs)):
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())   # .item() converts single-elem tensor to scalar
#if MAIN:
#    line(
#        loss_list, 
#        yaxis_range=[0, max(loss_list) + 0.1],
#        labels={"x": "Num batches seen", "y": "Cross entropy loss"}, 
#        title="ConvNet training on MNIST (cross entropy loss)",
#        width=700
#    )
#%%
if MAIN:
    probs = model(img_input).squeeze().softmax(-1).detach()

    bar(
        probs,
        x=range(1, 11),
        template="ggplot2",
        width=600,
        title="Classification probabilities", 
        labels={"x": "Digit", "y": "Probability"}, 
        text_auto='.2f',
        showlegend=False, 
        xaxis_tickmode="linear"
    )