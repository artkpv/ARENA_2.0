#%%
import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import einops
import wandb
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
import itertools
import random
from IPython.display import display
import wandb
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from typing import List, Union, Optional, Tuple, Callable, Dict
import typeguard
from functools import partial
import copy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm.notebook import tqdm
from dataclasses import dataclass
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning as pl
from rich import print as rprint
import pandas as pd

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part6_othellogpt"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from neel_plotly import scatter, line
import part6_othellogpt.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

t.set_grad_enabled(False)

MAIN = __name__ == "__main__"

# %%
alpha = "ABCDEFGH"

os.chdir(section_dir)

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

if not OTHELLO_ROOT.exists():
    !git clone https://github.com/likenneth/othello_world

sys.path.append(str(OTHELLO_MECHINT_ROOT))

from mech_interp_othello_utils import plot_board, plot_single_board, plot_board_log_probs, to_string, to_int, int_to_label, string_to_label, OthelloBoardState
# %%

cfg = HookedTransformerConfig(
    n_layers = 8,
    d_model = 512,
    d_head = 64,
    n_heads = 8,
    d_mlp = 2048,
    d_vocab = 61,
    n_ctx = 59,
    act_fn="gelu",
    normalization_type="LNPre",
    device=device,
)
model = HookedTransformer(cfg)
# %%

# Load board data as ints (i.e. 0 to 60)
board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)


# %%
full_linear_probe = t.load(OTHELLO_MECHINT_ROOT / "main_linear_probe.pth", map_location=device)

rows = 8
cols = 8 
options = 3
assert full_linear_probe.shape == (3, cfg.d_model, rows, cols, options)
# %% 

def state_stack_to_one_hot(state_stack):
    '''
    Creates a tensor of shape (games, moves, rows=8, cols=8, options=3), where the [g, m, r, c, :]-th entry
    is a one-hot encoded vector for the state of game g at move m, at row r and column c. In other words, this
    vector equals (1, 0, 0) when the state is empty, (0, 1, 0) when the state is "their", and (0, 0, 1) when the
    state is "my".
    '''
    one_hot = t.zeros(
        state_stack.shape[0], # num games
        state_stack.shape[1], # num moves
        rows,
        cols,
        3, # the options: empty, white, or black
        device=state_stack.device,
        dtype=t.int,
    )
    one_hot[..., 0] = state_stack == 0 
    one_hot[..., 1] = state_stack == -1 
    one_hot[..., 2] = state_stack == 1 

    return one_hot

# %%
# 4️⃣ Training a Probe

# %%
@dataclass
class ProbeTrainingArgs():

    # Which layer, and which positions in a game sequence to probe
    layer: int = 6
    pos_start: int = 5
    pos_end: int = model.cfg.n_ctx - 5
    length: int = pos_end - pos_start
    alternating: Tensor = t.tensor([1 if i%2 == 0 else -1 for i in range(length)], device=device)

    # Game state (options are blank/mine/theirs)
    options: int = 3
    rows: int = 8
    cols: int = 8

    # Standard training hyperparams
    max_epochs: int = 8
    num_games: int = 50000

    # Hyperparams for optimizer
    batch_size: int = 256
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    wd: float = 0.01

    # Misc.
    probe_name: str = "main_linear_probe"

    # The modes are "black to play / odd moves", "white to play / even moves", and "all moves"
    modes = 3

    # Code to get randomly initialized probe
    def setup_linear_probe(self, model: HookedTransformer):
        linear_probe = t.randn(
            self.modes, model.cfg.d_model, self.rows, self.cols, self.options, requires_grad=False, device=device
        ) / np.sqrt(model.cfg.d_model)
        linear_probe.requires_grad = True
        return linear_probe
# %%
def seq_to_state_stack(str_moves):
    board = OthelloBoardState()
    states = []
    for move in str_moves:
        board.umpire(move)
        states.append(np.copy(board.state))
    states = np.stack(states, axis=0)
    return states

# %%
class LitLinearProbe(pl.LightningModule):
    def __init__(self, model: HookedTransformer, args: ProbeTrainingArgs):
        super().__init__()
        self.model = model
        self.args = args
        self.linear_probe = args.setup_linear_probe(model)
        pl.seed_everything(42, workers=True)

    def training_step(self, batch: Int[Tensor, "game_idx"], batch_idx: int) -> t.Tensor:
        games_int = board_seqs_int[batch.cpu()]
        games_str = board_seqs_string[batch.cpu()]
        state_stack = t.stack([t.tensor(seq_to_state_stack(game_str.tolist())) for game_str in games_str])
        state_stack = state_stack[:, self.args.pos_start: self.args.pos_end, :, :]
        state_stack_one_hot = state_stack_to_one_hot(state_stack).to(device)
        batch_size = self.args.batch_size
        game_len = self.args.length

        # games_int = tensor of game sequences, each of length 60
        # This is the input to our model
        assert isinstance(games_int, Int[Tensor, f"batch={batch_size} full_game_len=60"])

        # state_stack_one_hot = tensor of one-hot encoded states for each game
        # We'll multiply this by our probe's estimated log probs along the `options` dimension, to get probe's estimated log probs for the correct option
        assert isinstance(state_stack_one_hot, Int[Tensor, f"batch={batch_size} game_len={game_len} rows=8 cols=8 options=3"])

        with t.inference_mode():
            _, cache = self.model.run_with_cache(
                games_int[:, :-1].to(device),
                return_type=None,
                names_filter=lambda name: name.endswith("resid_post")
                )
            resid_post = cache['resid_post', self.args.layer][:, self.args.pos_start: self.args.pos_end]
        logits = einops.einsum(
            resid_post,
            self.linear_probe,
            'g move dm, mode dm r c o -> mode g move r c o'
        )
        log_probs = logits.log_softmax(dim=-1)

        pred = einops.reduce(
            log_probs * state_stack_one_hot,
            "modes batch pos rows cols options -> modes pos rows cols",
            "mean"
        ) * self.args.options # Multiply to correct for the mean over options

        #pred = (state_stack_one_hot * log_probs).sum(dim=-1)# -> mode g move r c
        #pred = pred.mean(dim=1)  * self.args.options  # -> mode move r c 

        loss_even = -pred[0, 0::2].mean(dim=0)  # -> r c 
        loss_odd = -pred[1, 1::2].mean(dim=0)  # -> r c
        loss_all = -pred[2, :].mean(dim=0)  # -> r c
        loss = (loss_even + loss_odd + loss_all).sum()
        self.log('loss', loss)
        print(f"{loss=}")
        return loss
        

    def train_dataloader(self):
        '''
        Returns `games_int` and `state_stack_one_hot` tensors.
        '''
        n_indices = self.args.num_games - (self.args.num_games % self.args.batch_size)
        full_train_indices = t.randperm(self.args.num_games)[:n_indices]
        full_train_indices = einops.rearrange(full_train_indices, "(batch_idx game_idx) -> batch_idx game_idx", game_idx=self.args.batch_size)
        return full_train_indices

    def configure_optimizers(self):
        optimizer = t.optim.AdamW([self.linear_probe], lr=self.args.lr, betas=self.args.betas, weight_decay=self.args.wd)
        return optimizer

# %%
# Create the model & training system
args = ProbeTrainingArgs()
litmodel = LitLinearProbe(model, args)

# You can choose either logger
logger = CSVLogger(save_dir=os.getcwd() + "/logs", name=args.probe_name)
# logger = WandbLogger(save_dir=os.getcwd() + "/logs", project=args.probe_name)

# Train the model
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    logger=logger,
    log_every_n_steps=1,
)
trainer.fit(model=litmodel)

# %%

num_games = 50
focus_games_int = board_seqs_int[:num_games]
focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))

black_to_play_index = 0
white_to_play_index = 1
blank_index = 0
their_index = 1
my_index = 2

# Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")
my_linear_probe = t.zeros(cfg.d_model, rows, cols, options, device=device)
my_linear_probe[..., blank_index] = 0.5 * (litmodel.linear_probe[black_to_play_index, ..., 0] + litmodel.linear_probe[white_to_play_index, ..., 0])
my_linear_probe[..., their_index] = 0.5 * (litmodel.linear_probe[black_to_play_index, ..., 1] + litmodel.linear_probe[white_to_play_index, ..., 2])
my_linear_probe[..., my_index] = 0.5 * (litmodel.linear_probe[black_to_play_index, ..., 2] + litmodel.linear_probe[white_to_play_index, ..., 1])

# Getting the probe's output, and then its predictions
probe_out = einops.einsum(
    focus_cache["resid_post", 6], my_linear_probe,
    "game move d_model, d_model row col options -> game move row col options"
)
probe_out_value = probe_out.argmax(dim=-1)

# Getting the correct answers in the odd cases

focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
alternating = np.array([-1 if i%2 == 0 else 1 for i in range(focus_games_int.shape[1])])
flipped_focus_states = focus_states * alternating[None, :, None, None]
focus_states_flipped_one_hot = state_stack_to_one_hot(t.tensor(flipped_focus_states))
focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)
correct_middle_odd_answers = (probe_out_value == focus_states_flipped_value[:, :-1])[:, 5:-5:2]
accuracies_odd = einops.reduce(correct_middle_odd_answers.float(), "game move row col -> row col", "mean")

# Getting the correct answers in all cases
correct_middle_answers = (probe_out_value == focus_states_flipped_value[:, :-1])[:, 5:-5]
accuracies = einops.reduce(correct_middle_answers.float(), "game move row col -> row col", "mean")

def plot_square_as_board(state, diverging_scale=True, **kwargs):
    '''Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0'''
    kwargs = {
        "y": [i for i in alpha],
        "x": [str(i) for i in range(8)],
        "color_continuous_scale": "RdBu" if diverging_scale else "Blues",
        "color_continuous_midpoint": 0. if diverging_scale else None,
        "aspect": "equal",
        **kwargs
    }
    imshow(state, **kwargs)

plot_square_as_board(
    1 - t.stack([accuracies_odd, accuracies], dim=0), 
    title="Average Error Rate of Linear Probe", 
    facet_col=0, facet_labels=["Black to Play moves", "All Moves"], 
    zmax=0.25, zmin=-0.25
)
# %%
