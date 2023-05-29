#%%
import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"; os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import webbrowser
from pprint import pp

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

# Add this to your workspace settings, so typechecker sees these modules:
# "python.analysis.extraPaths": ["${workspaceFolder}/chapter1_transformers/exercises"]

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == '__main__'


#%%
if MAIN:
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small", 
        fold_ln=False, 
        center_unembed=False, 
        center_writing_weights=False
    )

#%%
if MAIN:
    sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
    print(sorted_vocab[:20])
    print()
    print(sorted_vocab[250:270])
    print()
    print(sorted_vocab[990:1010])
    print()

    lengths = dict.fromkeys(range(3, 8), "")
    for tok, idx in sorted_vocab:
        if not lengths.get(len(tok), True):
            lengths[len(tok)] = tok

    for length, tok in lengths.items():
        print(f"{length}: {tok}")

    print(sorted_vocab[-20:])

    print(f'Len: {len(sorted_vocab)}')
# %%
if MAIN:
    print(reference_gpt2.to_str_tokens("Ralph"))
    print(reference_gpt2.to_str_tokens(" Ralph"))
    print(reference_gpt2.to_str_tokens(" ralph"))
    print(reference_gpt2.to_str_tokens("ralph"))

    print(reference_gpt2.to_str_tokens("Artem"))
    print(reference_gpt2.to_str_tokens("Artyom"))
    print(reference_gpt2.to_str_tokens(" Artem"))
    print(reference_gpt2.to_str_tokens(" Artyom"))
    print(reference_gpt2.to_str_tokens("artem"))
    print(reference_gpt2.to_str_tokens("artyom"))

# %%
if MAIN:
    print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000"))
    print(reference_gpt2.to_str_tokens("2+2=4"))
    print(reference_gpt2.to_str_tokens("2000000+2000000=4000000"))

# %%
if MAIN:
    reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
    tokens = reference_gpt2.to_tokens(reference_text).to(device)

    print('tokens')
    print(tokens)
    print(tokens.shape)
    pp(reference_gpt2.to_str_tokens(tokens))

    logits, cache = reference_gpt2.run_with_cache(tokens)
    print('logits')
    print(logits.shape)
    probs = logits.softmax(dim=-1)
    print(f'{probs.shape=}')

    most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
    print('next token')
    pp(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

    pp('NEXT:')
    next_token = logits[0, -1].argmax(dim=-1)
    next_char = reference_gpt2.to_string(next_token)
    print(repr(next_char))

    print('Another way:')
    print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

    for i in range(10):
        print(f"{tokens.shape[-1]+1}th char = {next_char!r}")
        # Define new input sequence, by appending the previously generated token
        tokens = t.cat([tokens, next_token[None, None]], dim=-1)
        # Pass our new sequence through the model, to get new output
        logits = reference_gpt2(tokens)
        # Get the predicted token at the end of our sequence
        next_token = logits[0, -1].argmax(dim=-1)
        # Decode and print the result
        next_char = reference_gpt2.to_string(next_token)
# %%
