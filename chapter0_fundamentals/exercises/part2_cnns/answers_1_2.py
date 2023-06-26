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

MAIN = __name__ == "__main__"

#%%
arr = np.load(section_dir / "numbers.npy")

_d = display_array_as_img

if MAIN:
	_d(arr[0])
# %%
if MAIN:
	_d(einops.rearrange(arr, "b c h w -> c h (b w)"))
# %%

if MAIN:
	_d(einops.repeat(arr[0], "c h w -> c (h2 h) w", h2=2))
# %%

if MAIN:
	_d(einops.repeat(arr[0:2], "b c h w -> c (b h) (b2 w)", b2=2))

# %%

if MAIN:
	_d(einops.repeat(arr[0], "c h w -> c (h 2) w"))


# %%
if MAIN:
	_d(einops.repeat(arr[0], "c h w -> c (h 2) w"))

# %%
if MAIN:
	_d(einops.rearrange(arr[0], "c h w -> h (c w)"))
# %%
if MAIN:
	_d(einops.rearrange(arr, "(b b2) c h w -> c (b2 h) (b w)", b2=2))
	_d(einops.rearrange(arr, "(b2 b) c h w -> c (b2 h) (b w)", b2=2))
# %%
if MAIN:
	_d(einops.reduce(arr, "b c h w -> h (b w)", 'max'))
	_d(einops.reduce(arr, "b c h w -> h (b w)", 'min'))
# %%
if MAIN:
	_d(einops.reduce(arr, "b c h w -> h w", 'min'))
	_d(einops.reduce(arr, "b c h w -> h w", 'max'))
#%%
if MAIN:
	_d(einops.rearrange(arr[1], "c h w -> c w h"))
# %%
if MAIN:
	_d(einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2))
# %%

_s = einops.einsum

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return _s(mat, 'i i -> ')

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return _s(mat, vec, 'i j, j -> i')

def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return _s(mat1, mat2, 'i k, k j -> i j')

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return _s(vec1, vec2, 'i,i ->')

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return _s(vec1, vec2, 'i , j -> i j')


if MAIN:
    tests.test_einsum_trace(einsum_trace)
    tests.test_einsum_mv(einsum_mv)
    tests.test_einsum_mm(einsum_mm)
    tests.test_einsum_inner(einsum_inner)
    tests.test_einsum_outer(einsum_outer)
# %%

if MAIN:
    test_input = t.tensor(
        [[0, 1, 2, 3, 4], 
        [5, 6, 7, 8, 9], 
        [10, 11, 12, 13, 14], 
        [15, 16, 17, 18, 19]], dtype=t.float
    )
    display(test_input.stride())
# %%

if MAIN:
    TestCase = namedtuple("TestCase", ["output", "size", "stride"])

    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]), 
            size=(4,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 2], [5, 7]]), 
            size=(2,2),
            stride=(5,2),
        ),

        TestCase(
            output=t.tensor([0, 1, 2, 3, 4]),
            size=(5,),
            stride=(1,),
        ),

        TestCase(
            output=t.tensor([0, 5, 10, 15]),
            size=(4,),
            stride=(5,),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [5, 6, 7]
            ]), 
            size=(2,3),
            stride=(5,1),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [10, 11, 12]
            ]), 
            size=(2,3),
            stride=(10,1),
        ),

        TestCase(
            output=t.tensor([
                [0, 0, 0], 
                [11, 11, 11]
            ]), 
            size=(2,3),
            stride=(11,0),
        ),

        TestCase(
            output=t.tensor([0, 6, 12, 18]), 
            size=(4,),
            stride=(6,),
        ),
    ]

    for (i, test_case) in enumerate(test_cases):
        if (test_case.size is None) or (test_case.stride is None):
            print(f"Test {i} failed: attempt missing.")
        else:
            actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
            if (test_case.output != actual).any():
                print(f"Test {i} failed:")
                print(f"Expected: {test_case.output}")
                print(f"Actual: {actual}\n")
            else:
                print(f"Test {i} passed!\n")

# %%
def as_strided_trace(mat: Float[Tensor, "i j"]) -> Float[Tensor, ""]:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    n = mat.shape[0]
    return mat.as_strided(size=(n,), stride=(n + 1,)).sum()


if MAIN:
    tests.test_trace(as_strided_trace)
# %%
def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    n, m = mat.shape
    k = vec.shape[0]
    orig_stride = vec.stride(0)
    assert m == k, f'{m} == {k}'
    vec_s = t.as_strided(
        vec,
        size=(n,m),
        stride=(0,orig_stride)
    )
    res = (
        mat * vec_s
    ).sum(dim=1).T
    return res



if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)

# %%
def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    n,m = matA.shape
    k,p = matB.shape
    assert m == k, f'{m} == {k}'
    matAext = t.as_strided(
        matA,
        size=(n, m, p),
        stride=(matA.stride(0), matA.stride(1), 0)
    )
    matBext = t.as_strided(
        matB,
        size=(n, m, p),
        stride=(0, matB.stride(0), matB.stride(1))
    )
    return (matAext * matBext).sum(dim=1)

if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)
# %%
