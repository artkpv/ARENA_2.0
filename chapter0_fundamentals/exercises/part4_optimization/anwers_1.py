#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from IPython.display import display, HTML

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_optimization"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from plotly_utils import bar, imshow
from part3_resnets.solutions import IMAGENET_TRANSFORM, get_resnet_for_feature_extraction, plot_train_loss_and_test_accuracy_from_metrics
from part4_optimization.utils import plot_fn, plot_fn_with_points
import part4_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%

def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


if MAIN:
    plot_fn(pathological_curve_loss)

# %%
def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    out = []
    optimizer = t.optim.SGD([xy], lr=lr, momentum=momentum)
    while n_iters > 0:
        n_iters -= 1
        out.append(xy.detach().clone())
        optimizer.zero_grad()
        loss = fn(xy[0], xy[1])
        loss.backward()
        optimizer.step()
    return t.vstack(out)

#%%
if MAIN:
    points = []

    optimizer_list = [
        (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
        (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'])

        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)
# %%

class SGD:
    def __init__(
        self, 
        params: Iterable[t.nn.parameter.Parameter], 
        lr: float, 
        momentum: float = 0.0, 
        weight_decay: float = 0.0
    ):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        '''
        self.params = list(params) # turn params into a list (because it might be a generator)
        self.t = 0
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        grads = [p.grad for p in self.params]
        if self.weight_decay > 0:
            for g, p in zip(grads, self.params):
                g += self.weight_decay * p
        if self.momentum > 0:
            if self.t > 0:
                for i, (b, g) in enumerate(zip(self.b, grads)):
                    b *= self.momentum
                    b += g
                    grads[i] = b
            else:
                self.b = grads
        for p, g in zip(self.params, grads):
            p -= self.lr * g
        self.t += 1
        return self.params

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"



if MAIN:
    tests.test_sgd(SGD)

# %%
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

        '''
        self.params = list(params)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.bs = [t.zeros_like(p) for p in self.params]
        self.vs = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (p, v, b) in enumerate(zip(self.params,self.vs,self.bs)):
            g = p.grad
            if self.weight_decay != 0:
                g = g + self.weight_decay * p
            v = v*self.alpha + (1 - self.alpha) * g.pow(2)
            self.vs[i] = v
            if self.momentum > 0:
                b = b*self.momentum + g/(v.sqrt() + self.eps)
                self.bs[i] = b
                p -= self.lr * b
            else:
                p -= self.lr * g/(v.sqrt() + self.eps)
        return self.params
            
    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"

if MAIN:
    tests.test_rmsprop(RMSprop)

# %%
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        self.params = list(params)
        self.weight_decay = weight_decay
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.vs = [t.zeros_like(p) for p in self.params]
        self.ms = [t.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        self.t += 1
        for i, (p, v, m) in enumerate(zip(self.params,self.vs,self.ms)):
            g = p.grad
            if self.weight_decay != 0:
                g = g + self.weight_decay * p
            m = m*self.beta1 + (1 - self.beta1) * g
            v = v*self.beta2 + (1 - self.beta2) * g.pow(2)
            self.vs[i] = v
            self.ms[i] = m
            m_hat = m / (1 - pow(self.beta1, self.t))
            v_hat = v / (1 - pow(self.beta2, self.t))
            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
        return self.params

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
    tests.test_adam(Adam)

# %%
