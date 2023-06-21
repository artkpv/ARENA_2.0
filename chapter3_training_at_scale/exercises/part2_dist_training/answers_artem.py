#%%
import sys
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import torch
from torch import distributed as dist
from torch.distributed import ReduceOp
from typing import List
from pathlib import Path
import gdown

libncclpath = Path('/tmp/libnccl.so.2.18.1')
if not libncclpath.exists():
    gdown.download("https://drive.google.com/file/d/1QgkqHSPDwQD-Z0K0-4CUhp8fW-X0hWds/view", '/tmp/libnccl.so.2.18.1', quiet=False, fuzzy=True)
imagenetzip = Path('/root/arena_artem/chapter3_training_at_scale/exercises/part2_dist_training/imagenet_38k.zip')
if not imagenetzip.exists():
    gdown.download("https://drive.google.com/file/d/1tqUv0OktQdarW8hUyHjqNnxDP1JyUdkq/view?usp=sharing", quiet=False, fuzzy=True)

# Make sure exercises are in the path
chapter = r"chapter3_training_at_scale"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part7_toy_models_of_superposition"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
from threading import Thread
from time import sleep

# Add to the global variable
def adder(amount, repeats):
    global value
    for _ in range(repeats):
        value += amount

# Subtract from the global variable
def subtractor(amount, repeats):
    global value
    for _ in range(repeats):
        value -= amount

def add_and_subtract():
    # Start a thread making additions
    adder_thread = Thread(target=adder, args=(1, 10))
    adder_thread.start()
    #sleep(0.000001) 
    # Start a thread making subtractions
    subtractor_thread = Thread(target=subtractor, args=(1, 1000000))
    subtractor_thread.start()
    # Wait for both threads to finish
    print('Waiting for threads to finish...')
    adder_thread.join()
    subtractor_thread.join()
    # Print the value
    print(f'Value: {value}')


if __name__ == '__main__':
    value = 0
    add_and_subtract()
# %%
from test import test_broadcast_naive

def broadcast_naive(tensor: torch.Tensor, src: int):
    if dist.get_rank() == src:
        dist.broadcast(tensor, src)
        #for d in range(dist.get_world_size()):
        #    if d != src:
        #        dist.send(tensor, d)
    else:
        dist.recv(tensor, src)

if __name__ == '__main__':
    test_broadcast_naive(broadcast_naive)
    print('pass')
# %%
from test import test_broadcast_tree

def broadcast_tree(tensor: torch.Tensor, src: int):
    r = dist.get_rank()
    ws = dist.get_world_size()
    if r < src:
        dist.recv(tensor, r+1)
        if 0 <= r - 1:
            dist.send(tensor, r - 1)
    elif src < r:
        dist.recv(tensor, r-1)
        if r + 1 < ws:
            dist.send(tensor, r + 1)
    else:
        if 0 <= r - 1:
            dist.send(tensor, r - 1)
        if r + 1 < ws:
            dist.send(tensor, r + 1)

if __name__ == '__main__':
    test_broadcast_tree(broadcast_tree)
    print('pass')

# %%
from test import test_broadcast_ring

def broadcast_ring(tensor: torch.Tensor, src: int):
    r = dist.get_rank()
    ws = dist.get_world_size()
    if r != src:
        dist.recv(tensor, (ws+r-1)%ws)
    if (r+1)%ws != src:
        dist.send(tensor, (r+1)%ws)
    

if __name__ == '__main__':
    test_broadcast_ring(broadcast_ring)
    print('pass')
# %%
from test import test_reduce_naive

def reduce_naive(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    r = dist.get_rank()
    ws = dist.get_world_size()
    if r == dst:
        for i in range(ws):
            if i != r:
                received = torch.empty_like(tensor)
                dist.recv(received, i)
                if op == ReduceOp.SUM:
                    tensor.add_(received)
                elif op == ReduceOp.PRODUCT:
                    tensor.mul_(received)
                elif op == ReduceOp.MAX:
                    tensor = torch.max(tensor, received)
                elif op == ReduceOp.MIN:
                    tensor = torch.min(tensor, received)
                else:
                    raise NotImplementedError(f'op {op} not implemented')
    else:
        dist.send(tensor, dst)

if __name__ == '__main__':
    test_reduce_naive(reduce_naive)
    print('pass')
# %%
from test import test_reduce_tree

def reduce_tree(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    # SOLUTION
    curr_mult = dist.get_world_size() / 2
    rank_shifted = lambda: (dist.get_rank() - dst) % dist.get_world_size()
    while curr_mult >= 1:
        if rank_shifted() < curr_mult:
            buff = torch.empty_like(tensor)
            dist.recv(buff, (dist.get_rank() + curr_mult) % dist.get_world_size())
            if op == ReduceOp.SUM:
                tensor += buff
            elif op == ReduceOp.PRODUCT:
                tensor *= buff
            elif op == ReduceOp.MAX:
                tensor = torch.max(tensor, buff)
            elif op == ReduceOp.MIN:
                tensor = torch.min(tensor, buff)
            else:
                raise NotImplementedError(f'op {op} not implemented')
        elif rank_shifted() < curr_mult * 2:
            dist.send(tensor, (dist.get_rank() - curr_mult) % dist.get_world_size())
        curr_mult /= 2
    dist.barrier()

def _reduce(received, target, op):
    if op == ReduceOp.SUM:
        target.add_(received)
    elif op == ReduceOp.PRODUCT:
        target.mul_(received)
    elif op == ReduceOp.MAX:
        target = torch.max(target, received)
    elif op == ReduceOp.MIN:
        target = torch.min(target, received)
    else:
        raise NotImplementedError(f'op {op} not implemented')
    return target

def reduce_tree_WRONG(tensor: torch.Tensor, dst: int, op=ReduceOp.SUM):
    r = dist.get_rank()
    ws = dist.get_world_size()
    if r < dst:
        if 0 <= r-1:
            received = torch.empty_like(tensor)
            dist.recv(received, r-1)
            tensor = _reduce(received, tensor, op)
        dist.send(tensor, r+1)
    elif dst < r:
        if r+1 < ws:
            received = torch.empty_like(tensor)
            dist.recv(received, r+1)
            tensor = _reduce(received, tensor, op)
        dist.send(tensor, r-1)
    else:
        assert r == dst
        for i in (r-1, r+1):
            if 0 <= i < ws:
                received = torch.empty_like(tensor)
                dist.recv(received, i)
                tensor = _reduce(received, tensor, op)


if __name__ == '__main__':
    test_reduce_tree(reduce_tree)
    print('pass')

# %%
from test import test_allreduce_naive

def allreduce_naive(tensor: torch.Tensor, op=ReduceOp.SUM):
    r = dist.get_rank()
    root = 0
    if r != root:
        dist.send(tensor, root)
    dist.barrier()
    if r == root:
        ws = dist.get_world_size()
        for s in range(ws):
            if s != r:
                received = torch.empty_like(tensor)
                dist.recv(received, s)
                tensor = _reduce(received, tensor, op)
        dist.broadcast(tensor, r)
    dist.barrier()
    if r != root:
        dist.recv(tensor, root)

if __name__ == '__main__':
    test_allreduce_naive(allreduce_naive)
    print('pass')
# %%
from test import test_allreduce_butterfly
from math import ceil

def allreduce_butterfly(tensor: torch.Tensor, op=ReduceOp.SUM):
    i = dist.get_rank()
    ws = dist.get_world_size()
    span = ws
    while span > 1:
        span_start = span * (i // span)
        j = span_start + (i + span // 2) % span
        if 0 <= j < ws:
            dist.send(tensor, j)
            received = torch.empty_like(tensor)
            dist.recv(received, j)
            tensor = _reduce(received, tensor, op)
        dist.barrier()
        span = int(ceil(span / 2))

if __name__ == '__main__':
    test_allreduce_butterfly(allreduce_butterfly)
    print('pass')
# %%
