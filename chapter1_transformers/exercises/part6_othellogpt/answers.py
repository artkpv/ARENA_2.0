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

from pprint import pp
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
sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
# champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
model.load_state_dict(sd)
# %%
# An example input
sample_input = t.tensor([[
    20, 19, 18, 10,  2,  1, 27,  3, 41, 42, 34, 12,  4, 40, 11, 29, 43, 13, 48, 56, 
    33, 39, 22, 44, 24,  5, 46,  6, 32, 36, 51, 58, 52, 60, 21, 53, 26, 31, 37,  9,
    25, 38, 23, 50, 45, 17, 47, 28, 35, 30, 54, 16, 59, 49, 57, 14, 15, 55, 7
]]).to(device)

# The argmax of the output (ie the most likely next move from each position)
sample_output = t.tensor([[
    21, 41, 40, 34, 40, 41,  3, 11, 21, 43, 40, 21, 28, 50, 33, 50, 33,  5, 33,  5,
    52, 46, 14, 46, 14, 47, 38, 57, 36, 50, 38, 15, 28, 26, 28, 59, 50, 28, 14, 28, 
    28, 28, 28, 45, 28, 35, 15, 14, 30, 59, 49, 59, 15, 15, 14, 15,  8,  7,  8
]]).to(device)

assert (model(sample_input).argmax(dim=-1) == sample_output.to(device)).all()

# %%
os.chdir(section_dir)

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

if not OTHELLO_ROOT.exists():
    !git clone https://github.com/likenneth/othello_world

sys.path.append(str(OTHELLO_MECHINT_ROOT))

# %%
from mech_interp_othello_utils import plot_board, plot_single_board, plot_board_log_probs, to_string, to_int, int_to_label, string_to_label, OthelloBoardState
# %%
# Load board data as ints (i.e. 0 to 60)
board_seqs_int = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"), dtype=t.long)
# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
board_seqs_string = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"), dtype=t.long)

assert all([middle_sq not in board_seqs_string for middle_sq in [27, 28, 35, 36]])
'''
0 1 2 3  4 5 6 7
8 9 0 1  2 3 4 15
6 7 8 9  0 1 2 23
4 5 6 7  8 9 0 31 

2 3 4 5  6 7 8 39 
0 1 2 3  4 5 6 47
8 9 0 1  2 3 4 55
6 7 8 9  0 1 2 63
'''
assert board_seqs_int.max() == 60

num_games, length_of_game = board_seqs_int.shape
print("Number of games:", num_games)
print("Length of game:", length_of_game)
pp(board_seqs_int[10])
pp(board_seqs_string[10])

# %%
# Define possible indices (excluding the four center squares)
stoi_indices = [i for i in range(64) if i not in [27, 28, 35, 36]]

# Define our rows, and the function that converts an index into a (row, column) label, e.g. `E2`
alpha = "ABCDEFGH"

def to_board_label(i):
    return f"{alpha[i//8]}{i%8}"

# Get our list of board labels

board_labels = list(map(to_board_label, stoi_indices))
full_board_labels = list(map(to_board_label, range(64)))
print(board_labels)
print(full_board_labels)
# %%
moves_int = board_seqs_int[0, :30]

# This is implicitly converted to a batch of size 1
logits: Tensor = model(moves_int)
print("logits:", logits.shape)
print(logits)
# %%
logit_vec = logits[0, -1]
log_probs = logit_vec.log_softmax(-1)
# Remove the "pass" move (the zeroth vocab item)
log_probs = log_probs[1:]
assert len(log_probs)==60

# Set all cells to -13 by default, for a very negative log prob - this means the middle cells don't show up as mattering
temp_board_state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
temp_board_state.flatten()[stoi_indices] = log_probs
# %%
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


plot_square_as_board(temp_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title="Example Log Probs")
# %%
plot_single_board(int_to_label(moves_int))
# %%
num_games = 50
focus_games_int = board_seqs_int[:num_games]
focus_games_string = board_seqs_string[:num_games]
def one_hot(list_of_ints, num_classes=64):
    out = t.zeros((num_classes,), dtype=t.float32)
    out[list_of_ints] = 1.
    return out


focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
focus_valid_moves = t.zeros((num_games, 60, 64), dtype=t.float32)

for i in (range(num_games)):
    board = OthelloBoardState()
    for j in range(60):
        board.umpire(focus_games_string[i, j].item())
        focus_states[i, j] = board.state
        focus_valid_moves[i, j] = one_hot(board.get_valid_moves())

print("focus states:", focus_states.shape)
print("focus_valid_moves", tuple(focus_valid_moves.shape))
# %%
imshow(
    focus_states[0, :17],
    facet_col=0,
    facet_col_wrap=8,
    facet_labels=[f"Move {i}" for i in range(1, 17)],
    title="First 16 moves of first game",
    color_continuous_scale="Greys",
)
# %%
focus_logits, focus_cache = model.run_with_cache(focus_games_int[:, :-1].to(device))
# %%
print(focus_logits.shape)
print(focus_cache['pattern', 7].shape)
# %%
full_linear_probe = t.load(OTHELLO_MECHINT_ROOT / "main_linear_probe.pth", map_location=device)

rows = 8
cols = 8 
options = 3
assert full_linear_probe.shape == (3, cfg.d_model, rows, cols, options)

# %%
black_to_play_index = 0
white_to_play_index = 1
blank_index = 0
their_index = 1
my_index = 2

# Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")
linear_probe = t.zeros(cfg.d_model, rows, cols, options, device=device)
linear_probe[..., blank_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 0] + full_linear_probe[white_to_play_index, ..., 0])
linear_probe[..., their_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 1] + full_linear_probe[white_to_play_index, ..., 2])
linear_probe[..., my_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 2] + full_linear_probe[white_to_play_index, ..., 1])
# %%
layer = 6
game_index = 0
move = 29

def plot_probe_outputs(layer, game_index, move, **kwargs):
    residual_stream = focus_cache["resid_post", layer][game_index, move]
    # print("residual_stream", residual_stream.shape)
    probe_out = einops.einsum(residual_stream, linear_probe, "d_model, d_model row col options -> row col options")
    probabilities = probe_out.softmax(dim=-1)
    plot_square_as_board(probabilities, facet_col=2, facet_labels=["P(Empty)", "P(Their's)", "P(Mine)"], **kwargs)


plot_probe_outputs(layer, game_index, move, title="Example probe outputs after move 29 (black to play)")

plot_single_board(int_to_label(focus_games_int[game_index, :move+1]))
# %%
layer = 4
game_index = 0
move = 29

plot_probe_outputs(layer, game_index, move, title="Example probe outputs at layer 4 after move 29 (black to play)")
#%%
layer = 4
game_index = 0
move = 30

plot_probe_outputs(layer, game_index, move, title="Example probe outputs at layer 4 after move 30 (white to play)")

plot_single_board(focus_games_string[game_index, :31])
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

# We first convert the board states to be in terms of my (+1) and their (-1), rather than black and white

alternating = np.array([-1 if i%2 == 0 else 1 for i in range(focus_games_int.shape[1])])
flipped_focus_states = focus_states * alternating[None, :, None, None]

# We now convert to one-hot encoded vectors
focus_states_flipped_one_hot = state_stack_to_one_hot(t.tensor(flipped_focus_states))

# Take the argmax (i.e. the index of option empty/their/mine)
focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)
# %%
print(focus_states_flipped_value.shape)
# %%
probe_out = einops.einsum(
    focus_cache["resid_post", 6], linear_probe,
    "game move d_model, d_model row col options -> game move row col options"
)

probe_out_value = probe_out.argmax(dim=-1)
#$$
correct_middle_odd_answers = (probe_out_value.cpu() == focus_states_flipped_value[:, :-1])[:, 5:-5:2]
accuracies_odd = einops.reduce(correct_middle_odd_answers.float(), "game move row col -> row col", "mean")

correct_middle_answers = (probe_out_value.cpu() == focus_states_flipped_value[:, :-1])[:, 5:-5]
accuracies = einops.reduce(correct_middle_answers.float(), "game move row col -> row col", "mean")

plot_square_as_board(
    1 - t.stack([accuracies_odd, accuracies], dim=0),
    title="Average Error Rate of Linear Probe", 
    facet_col=0, facet_labels=["Black to Play moves", "All Moves"], 
    zmax=0.25, zmin=-0.25
)
# %%
#print((probe_out_value.cpu() == focus_states_flipped_value[:, :-1]).float().mean())
#pp(einops.reduce(correct_middle_odd_answers.float(), "game move row col -> row col", "mean"))
#pp(einops.reduce(correct_middle_answers.float(), "game move row col -> row col", "mean"))
# %%

odd_mode_B_minus_W = full_linear_probe[black_to_play_index, ..., 1] - full_linear_probe[black_to_play_index, ..., 2]
even_mode_B_minus_W = full_linear_probe[white_to_play_index, ..., 1] - full_linear_probe[white_to_play_index, ..., 2]
stacked = t.stack([odd_mode_B_minus_W, even_mode_B_minus_W], dim=0)
by_mode_cell = einops.rearrange(stacked, "mode d_model row col -> (mode row col) d_model")
by_mode_cell /= by_mode_cell.norm(dim=-1, keepdim=True)
cosine_similarities = einops.einsum(by_mode_cell, by_mode_cell, "x d_model, y d_model -> x y")
imshow(
    cosine_similarities,
    title="Cosine Sim of B-W Linear Probe Directions by Cell",
    x=[f"{L} (O)" for L in full_board_labels] + [f"{L} (E)" for L in full_board_labels],
    y=[f"{L} (O)" for L in full_board_labels] + [f"{L} (E)" for L in full_board_labels],
)
# %%
# YOUR CODE HERE - define `blank_probe` and `my_probe`
blank_probe = linear_probe[..., blank_index] - (linear_probe[..., my_index] + linear_probe[..., their_index]) / 2
my_probe = linear_probe[..., my_index] - linear_probe[..., their_index]
tests.test_my_probes(blank_probe, my_probe, linear_probe)
# %%
pos = 20
game_index = 0

# Plot board state
moves = focus_games_string[game_index, :pos+1]
plot_single_board(moves)

# Plot corresponding model predictions
state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
state.flatten()[stoi_indices] = focus_logits[game_index, pos].log_softmax(dim=-1)[1:]
plot_square_as_board(state, zmax=0, diverging_scale=False, title="Log probs")
# %%
cell_r = 5
cell_c = 4
print(f"Flipping the color of cell {'ABCDEFGH'[cell_r]}{cell_c}")

board = OthelloBoardState()
board.update(moves.tolist())
board_state = board.state.copy()
valid_moves = board.get_valid_moves()
flipped_board = copy.deepcopy(board)
flipped_board.state[cell_r, cell_c] *= -1
flipped_valid_moves = flipped_board.get_valid_moves()

newly_legal = [string_to_label(move) for move in flipped_valid_moves if move not in valid_moves]
newly_illegal = [string_to_label(move) for move in valid_moves if move not in flipped_valid_moves]
print("newly_legal", newly_legal)
print("newly_illegal", newly_illegal)
# %%
def apply_scale(
        resid: Float[Tensor, "batch=1 seq d_model"],
        flip_dir: Float[Tensor, "d_model"],
        scale: int,
        pos: int
    ):
    '''
    Returns a version of the residual stream, modified by the amount `scale` in the 
    direction `flip_dir` at the sequence position `pos`, in the way described above.
    '''
    flip_dir /= flip_dir.norm()
    dot_pr = resid[0, pos] @ flip_dir 
    resid[0, pos] -= (scale+1) * dot_pr * flip_dir  
    return resid

tests.test_apply_scale(apply_scale)


# %%
flip_dir = my_probe[:, cell_r, cell_c]

big_flipped_states_list = []
layer = 4
scales = [0, 1, 2, 4, 8, 16]

# Iterate through scales, generate a new facet plot for each possible scale
for scale in scales:

    # Hook function which will perform flipping in the "F4 flip direction"
    def flip_hook(resid: Float[Tensor, "batch=1 seq d_model"], hook: HookPoint):
        return apply_scale(resid, flip_dir, scale, pos)

    # Calculate the logits for the board state, with the `flip_hook` intervention
    # (note that we only need to use :pos+1 as input, because of causal attention)
    flipped_logits: Tensor = model.run_with_hooks(
        focus_games_int[game_index:game_index+1, :pos+1],
        fwd_hooks=[
            (utils.get_act_name("resid_post", layer), flip_hook),
        ]
    ).log_softmax(dim=-1)[0, pos]

    flip_state = t.zeros((64,), dtype=t.float32, device=device) - 10.
    flip_state[stoi_indices] = flipped_logits.log_softmax(dim=-1)[1:]
    big_flipped_states_list.append(flip_state)


flip_state_big = t.stack(big_flipped_states_list)
state_big = einops.repeat(state.flatten(), "d -> b d", b=6)
color = t.zeros((len(scales), 64)).to(device) + 0.2
for s in newly_legal:
    color[:, to_string(s)] = 1
for s in newly_illegal:
    color[:, to_string(s)] = -1

scatter(
    y=state_big, 
    x=flip_state_big, 
    title=f"Original vs Flipped {string_to_label(8*cell_r+cell_c)} at Layer {layer}", 
    # labels={"x": "Flipped", "y": "Original"}, 
    xaxis="Flipped", 
    yaxis="Original", 

    hover=[f"{r}{c}" for r in "ABCDEFGH" for c in range(8)], 
    facet_col=0, facet_labels=[f"Translate by {i}x" for i in scales], 
    color=color, color_name="Newly Legal", color_continuous_scale="Geyser"
)
# %%
# Part 2️⃣. Looking for modular circuits
##########################################

# %%
game_index = 1
move = 20
layer = 6

plot_single_board(focus_games_string[game_index, :move+1])
plot_probe_outputs(layer, game_index, move)
# %%
def plot_contributions(contributions, component: str, probe_desc: str="my vs their"):
    imshow(
        contributions,
        facet_col=0,
        y=list("ABCDEFGH"),
        facet_labels=[f"Layer {i}" for i in range(7)],
        title=f"{component} Layer Contributions to {probe_desc} (Game {game_index} Move {move})",
        aspect="equal",
        width=1400,
        height=350
    )

def calculate_attn_and_mlp_probe_score_contributions(
    focus_cache: ActivationCache, 
    my_probe: Float[Tensor, "d_model rows cols"],
    layer: int,
    game_index: int, 
    move: int
) -> Tuple[Float[Tensor, "layers rows cols"], Float[Tensor, "layers rows cols"]]:
    attn_contributions = t.zeros((layer+1, rows, cols), dtype=t.float32, device=device)
    mlp_contributions = t.zeros((layer+1, rows, cols), dtype=t.float32, device=device)
    for l in range(layer+1):
        attn_contributions[l] = einops.einsum(
            focus_cache["attn_out", l][game_index, move],
            my_probe,
            "d_model, d_model row col -> row col"
        )
        mlp_contributions[l] = einops.einsum(
            focus_cache["mlp_out", l][game_index, move],
            my_probe,
            "d_model, d_model row col -> row col"
        )
    return attn_contributions, mlp_contributions

attn_contributions, mlp_contributions = calculate_attn_and_mlp_probe_score_contributions(
    focus_cache, my_probe, layer, game_index, move)

plot_contributions(attn_contributions, "Attention")
plot_contributions(mlp_contributions, "MLP")
# %%
def calculate_accumulated_probe_score(
    focus_cache: ActivationCache, 
    my_probe: Float[Tensor, "d_model rows cols"],
    layer: int,
    game_index: int, 
    move: int
) -> Float[Tensor, "rows cols"]:
    return einops.einsum(
        focus_cache["resid_post", layer][game_index, move],
        my_probe,
        "d_model, d_model row col -> row col"
    )


overall_contribution = calculate_accumulated_probe_score(focus_cache, my_probe, layer, game_index, move)

imshow(
    overall_contribution, 
    title=f"Overall Probe Score after Layer {layer} for<br>my vs their (Game {game_index} Move {move})",
)
# %%

attn_contributions, mlp_contributions = calculate_attn_and_mlp_probe_score_contributions(
    focus_cache, blank_probe, layer, game_index, move)

plot_contributions(attn_contributions, "Attention", probe_desc="blank vs non-blank")
plot_contributions(mlp_contributions, "MLP", probe_desc="blank vs non-blank")

overall_contribution = calculate_accumulated_probe_score(focus_cache, blank_probe, layer, game_index, move)

imshow(
    overall_contribution, 
    title=f"Overall Probe Score after Layer {layer} for<br>blank vs non-blank (Game {game_index} Move {move})",
)
# %%
# Scale the probes down to be unit norm per cell
blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
# Set the center blank probes to 0, since they're never blank so the probe is meaningless
blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.
# %%
def get_w_in(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    '''
    Returns the input weights for the given neuron.

    If normalize is True, the weights are normalized to unit norm.
    '''
    w = model.W_in[layer, :, neuron].detach().clone()
    if normalize:
        w /= w.norm()
    return w
    

def get_w_out(
    model: HookedTransformer,
    layer: int,
    neuron: int,
    normalize: bool = False,
) -> Float[Tensor, "d_model"]:
    '''
    Returns the input weights for the given neuron.

    If normalize is True, the weights are normalized to unit norm.
    '''
    w = model.W_out[layer, neuron].detach().clone()
    if normalize:
        w /= w.norm()
    return w


def calculate_neuron_input_weights(
    model: HookedTransformer, 
    probe: Float[Tensor, "d_model row col"], 
    layer: int, 
    neuron: int
) -> Float[Tensor, "rows cols"]:
    '''
    Returns tensor of the input weights for the given neuron, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    '''
    neuron_w = get_w_in(model, layer, neuron, normalize=True)
    return einops.einsum(
        neuron_w,
        probe,
        "d_model, d_model row col -> row col"
    )


def calculate_neuron_output_weights(
    model: HookedTransformer, 
    probe: Float[Tensor, "d_model row col"], 
    layer: int, 
    neuron: int
) -> Float[Tensor, "rows cols"]:
    '''
    Returns tensor of the output weights for the given neuron, at each square on the board,
    projected along the corresponding probe directions.

    Assume probe directions are normalized. You should also normalize the model weights.
    '''
    neuron_w = get_w_out(model, layer, neuron, normalize=True)
    return einops.einsum(
        neuron_w,
        probe,
        "d_model, d_model row col -> row col"
    )


tests.test_calculate_neuron_input_weights(calculate_neuron_input_weights, model)
tests.test_calculate_neuron_output_weights(calculate_neuron_output_weights, model)
# %%
layer = 5
neuron = 1393

w_in_L5N1393_blank = calculate_neuron_input_weights(model, blank_probe_normalised, layer, neuron)
w_in_L5N1393_my = calculate_neuron_input_weights(model, my_probe_normalised, layer, neuron)

imshow(
    t.stack([w_in_L5N1393_blank, w_in_L5N1393_my]),
    facet_col=0,
    y=[i for i in "ABCDEFGH"],
    title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
    facet_labels=["Blank In", "My In"],
    width=750,
)

#%%
w_out_L5N1393_blank = calculate_neuron_output_weights(model, blank_probe_normalised, layer, neuron)
w_out_L5N1393_my = calculate_neuron_output_weights(model, my_probe_normalised, layer, neuron)

imshow(
    t.stack([w_out_L5N1393_blank, w_out_L5N1393_my]),
    facet_col=0,
    y=[i for i in "ABCDEFGH"],
    title=f"Output weights in terms of the probe for neuron L{layer}N{neuron}",
    facet_labels=["Blank Out", "My Out"],
    width=750,
)

# %%
w_in_L5N1393 = get_w_in(model, layer, neuron, normalize=True)
w_out_L5N1393 = get_w_out(model, layer, neuron, normalize=True)

U, S, Vh = t.svd(t.cat([
    my_probe.reshape(cfg.d_model, 64),
    blank_probe.reshape(cfg.d_model, 64)
], dim=1))

# Remove the final four dimensions of U, as the 4 center cells are never blank and so the blank probe is meaningless there
probe_space_basis = U[:, :-4]

print("Fraction of input weights in probe basis:", (w_in_L5N1393 @ probe_space_basis).norm().item()**2)
print("Fraction of output weights in probe basis:", (w_out_L5N1393 @ probe_space_basis).norm().item()**2)

#%%
def kurtosis(tensor: Tensor, reduced_axes, fisher=True):
    '''
    Computes the kurtosis of a tensor over specified dimensions.
    '''
    return (((tensor - tensor.mean(dim=reduced_axes, keepdim=True)) / tensor.std(dim=reduced_axes, keepdim=True))**4).mean(dim=reduced_axes, keepdim=False) - fisher*3

# %%
layer = 3
top_layer_3_neurons = focus_cache["post", layer][:, 3:-3].std(dim=[0, 1]).argsort(descending=True)[:10]
#top_layer_3_neurons = einops.reduce(focus_cache["post", layer][:, 3:-3], "game move neuron -> neuron", reduction=kurtosis).argsort(descending=True)[:10]

heatmaps_blank = []
heatmaps_my = []

for neuron in top_layer_3_neurons:
    neuron = neuron.item()
    heatmaps_blank.append(calculate_neuron_output_weights(model, blank_probe_normalised, layer, neuron))
    heatmaps_my.append(calculate_neuron_output_weights(model, my_probe_normalised, layer, neuron))

imshow(
    t.stack(heatmaps_blank),
    facet_col=0,
    y=[i for i in "ABCDEFGH"],
    title=f"Cosine sim of Output weights and the 'blank color' probe for top layer {layer} neurons",
    facet_labels=[f"L3N{n.item()}" for n in top_layer_3_neurons],
    width=1600, height=300,
)

imshow(
    t.stack(heatmaps_my),
    facet_col=0,
    y=[i for i in "ABCDEFGH"],
    title=f"Cosine sim of Output weights and the 'my color' probe for top layer {layer} neurons",
    facet_labels=[f"L3N{n.item()}" for n in top_layer_3_neurons],
    width=1600, height=300,
)
# %%
layer = 4
top_layer_4_neurons = focus_cache["post", layer][:, 3:-3].std(dim=[0, 1]).argsort(descending=True)[:10]

heatmaps_blank = []
heatmaps_my = []

for neuron in top_layer_4_neurons:
    neuron = neuron.item()
    heatmaps_blank.append(calculate_neuron_output_weights(model, blank_probe_normalised, layer, neuron))
    heatmaps_my.append(calculate_neuron_output_weights(model, my_probe_normalised, layer, neuron))

imshow(
    t.stack(heatmaps_blank),
    facet_col=0,
    y=[i for i in "ABCDEFGH"],
    title=f"Cosine sim of Output weights and the blank color probe for top layer 4 neurons",
    facet_labels=[f"L4N{n.item()}" for n in top_layer_4_neurons],
    width=1600, height=300,
)

imshow(
    t.stack(heatmaps_my),
    facet_col=0,
    y=[i for i in "ABCDEFGH"],
    title=f"Cosine sim of Output weights and the my color probe for top layer 4 neurons",
    facet_labels=[f"L4N{n.item()}" for n in top_layer_4_neurons],
    width=1600, height=300,
)
# %%
'''
> A cell can only be legal to play in if it is blank (obviously). Since calculating blankness is easy (you just check whether a move was played), the model should be able to do this in a single layer - this is what we see above.
> Question - if this is true, then what observation should we expect when we compare the neuron output weights to the unembedding weights?

This neuron ouput weights should have high contribution to the unembedding weights because blankness shouldn't change.
'''
# %% 
layer = 4
top_layer_4_neurons = focus_cache["post", layer][:, 3:-3].std(dim=[0, 1]).argsort(descending=True)[:10]
W_U_norm = model.W_U / model.W_U.norm(dim=0, keepdim=True)
W_U_norm = W_U_norm[:, 1:] # Get rid of the passing/dummy first element
heatmaps_unembed = []

for neuron in top_layer_4_neurons:
    neuron = neuron.item()
    w_out = get_w_out(model, layer, neuron)
    # Fill in the `state` tensor with cosine sims, while skipping the middle 4 squares
    state = t.zeros((8, 8), device=device)
    state.flatten()[stoi_indices] = w_out @ W_U_norm
    heatmaps_unembed.append(state)

imshow(
    t.stack(heatmaps_unembed),
    facet_col=0,
    y=[i for i in "ABCDEFGH"],
    title=f"Cosine sim of Output weights and the unembed for top layer 4 neurons",
    facet_labels=[f"L4N{n.item()}" for n in top_layer_4_neurons],
    width=1600, height=300,
)

#%%
# Activation Patching
#####################

# %%
game_index = 4
move = 20

plot_single_board(focus_games_string[game_index, :move+1], title="Original Game (black plays E0)")
plot_single_board(focus_games_string[game_index, :move].tolist()+[16], title="Corrupted Game (blank plays C0)")
# %%
clean_input = focus_games_int[game_index, :move+1].clone()
corrupted_input = focus_games_int[game_index, :move+1].clone()
corrupted_input[-1] = to_int("C0")
print("Clean:     ", ", ".join(int_to_label(corrupted_input)))
print("Corrupted: ", ", ".join(int_to_label(clean_input)))
# %%
clean_logits, clean_cache = model.run_with_cache(clean_input)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_input)

clean_log_probs = clean_logits.log_softmax(dim=-1)
corrupted_log_probs = corrupted_logits.log_softmax(dim=-1)
# %%
f0_index = to_int("F0")
clean_f0_log_prob = clean_log_probs[0, -1, f0_index]
corrupted_f0_log_prob = corrupted_log_probs[0, -1, f0_index]

print("Clean log prob", clean_f0_log_prob.item())
print("Corrupted log prob", corrupted_f0_log_prob.item(), "\n")

def patching_metric(patched_logits: Float[Tensor, "batch=1 seq=21 d_vocab=61"]):
    '''
    Function of patched logits, calibrated so that it equals 0 when performance is 
    same as on corrupted input, and 1 when performance is same as on clean input.

    Should be linear function of the logits for the F0 token at the final move.
    '''
    log_prob = patched_logits.log_softmax(dim=-1)[0, -1, f0_index] 
    return (log_prob - corrupted_f0_log_prob) / (clean_f0_log_prob - corrupted_f0_log_prob)


tests.test_patching_metric(patching_metric, clean_log_probs, corrupted_log_probs)
# %%
def patch_final_move_output(
    activation: Float[Tensor, "batch seq d_model"], 
    hook: HookPoint,
    clean_cache: ActivationCache,
) -> Float[Tensor, "batch seq d_model"]:
    '''
    Hook function which patches activations at the final sequence position.

    Note, we only need to patch in the final sequence position, because the
    prior moves in the clean and corrupted input are identical (and this is
    an autoregressive model).
    '''
    return clean_cache[hook.name]
    #activation[0,-1,:] =  clean_cache[hook.name][0, -1, :]
    #return activation

def get_act_patch_resid_pre(
    model: HookedTransformer, 
    corrupted_input: Float[Tensor, "batch pos"], 
    clean_cache: ActivationCache, 
    patching_metric: Callable[[Float[Tensor, "batch seq d_model"]], Float[Tensor, ""]]
) -> Float[Tensor, "2 n_layers"]:
    '''
    Returns an array of results, corresponding to the results of patching at
    each (attn_out, mlp_out) for all layers in the model.
    '''
    res = t.empty((2, model.cfg.n_layers), device=device)
    for act_i, act_name in ((0, 'attn_out'), (1, 'mlp_out')):
        for layer in range(model.cfg.n_layers):
            model.reset_hooks()
            out = model.run_with_hooks(corrupted_input, fwd_hooks=[
                (utils.get_act_name(act_name, layer), lambda resid, hook: patch_final_move_output(resid, hook, clean_cache)),
            ])
            res[act_i, layer] = patching_metric(out)
    return res

patching_results = get_act_patch_resid_pre(model, corrupted_input, clean_cache, patching_metric)

line(patching_results, title="Layer Output Patching Effect on F0 Log Prob", line_labels=["attn", "mlp"], width=750)
# %%
# Part 3️⃣ Neuron Interpretability: A Deep Dive
# %%
layer = 5
neuron = 1393

w_out = get_w_out(model, layer, neuron, normalize=False)
state = t.zeros(8, 8, device=device)
state.flatten()[stoi_indices] = w_out @ model.W_U[:, 1:]
plot_square_as_board(state, title=f"Output weights of Neuron L{layer}N{neuron} in the output logit basis", width=600)
# %%
c0 = model.W_U[:, to_int("C0")].detach()
d1 = model.W_U[:, to_int("D1")].detach()
print('Similarity of C0 and D1:', (c0 @ d1).item())
# %%
w_out /= w_out.norm()
U, S, Vh = t.svd(model.W_U[:,1:])
W_U_space_basis = U[:, :-4]

print("Fraction of neuron weights in W_U basis:", (w_out @ probe_space_basis).norm().item()**2)
# %%
neuron_acts = focus_cache["post", layer, "mlp"][:, :, neuron]

imshow(
    neuron_acts,
    title=f"L{layer}N{neuron} Activations over 50 games",
    labels={"x": "Move", "y": "Game"},
    aspect="auto",
    width=900
)
# %%
game = 18
# for move in range(60):
#     if neuron_acts[game, move].item() > 0:
#         temp_board_state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.
#         temp_board_state.flatten()[stoi_indices] = focus_logits[game, move].log_softmax(dim=-1)[1:]
#         plot_square_as_board(temp_board_state.reshape(8, 8), zmax=0, diverging_scale=False, title=f"Game {game}, move {move}")
# Result of the above is that it prints boards for even moves from 14 to 42 where C0 is activated. 
# %%
game = 18
moves=slice(30,46)
imshow(
    focus_states[game, moves],
    facet_col=0,
    facet_col_wrap=5,
    y=list("ABCDEFGH"),
    facet_labels=[f"Move {i}" for i in list(range(60))[moves]],
    title=f"First moves of {game} game",
    color_continuous_scale="Greys",
    coloraxis_showscale=False,
    width=1000,
    height=1000,
)
# %%
# Max Activating Datasets
# %%
focus_states_flipped_pm1 = t.zeros_like(focus_states_flipped_value, device=device)
focus_states_flipped_pm1[focus_states_flipped_value==2] = -1.
focus_states_flipped_pm1[focus_states_flipped_value==1] = 1.

def _investigate_the_neuron_max_activating_ds():
    global focus_states_flipped_pm1
    print(f"{neuron_acts.shape=}")
    print(f"{focus_states_flipped_value.shape=}")
    top_moves = neuron_acts > neuron_acts.quantile(0.99)

    #focus_states_flipped_value = focus_states_flipped_value.to(device)
    board_state_at_top_moves = t.stack([
        (focus_states_flipped_value == 2)[:, :-1][top_moves].float().mean(0),  # Mine
        (focus_states_flipped_value == 1)[:, :-1][top_moves].float().mean(0),  # Theirs
        (focus_states_flipped_value == 0)[:, :-1][top_moves].float().mean(0)   # Blank
    ])

    plot_square_as_board(
        board_state_at_top_moves, 
        facet_col=0,
        facet_labels=["Mine", "Theirs", "Blank"],
        title=f"Aggregated top 30 moves for neuron L{layer}N{neuron}", 
    )

    board_state_at_top_moves = focus_states_flipped_pm1[:, :-1][top_moves].float().mean(0)

    plot_square_as_board(
        board_state_at_top_moves, 
        title=f"Aggregated top 30 moves for neuron L{layer}N{neuron} (1 = theirs, -1 = mine)",
    )

_investigate_the_neuron_max_activating_ds()

# %%

# %%
# Exercise - Investigating more neurons
# Your code here - investigate the top 10 neurons by std dev of activations, see what you can find!

def _investigate_more_neurons():
    layer = 5
    num = 10
    
    focus = focus_cache["post", layer]  # game move dim
    #focus_cache_mlp = einops.einsum(focus, 'g m d -> d')
    #print(f"{focus_cache_mlp.shape=}")
    #top_neurons = einops.reduce(focus, 'g m d -> d', 'mean').topk(num, dim=-1).indices
    #print(top_neurons)
    #top_neurons = einops.reduce(focus, 'g m d -> d', 'sum').topk(num, dim=-1).indices
    #print(top_neurons)
    top_neurons = focus.std(dim=(0,1)).topk(num).indices
    print(top_neurons)

    # W_out to W_U : 
    w_out = model.W_out[layer, top_neurons].detach().clone()
    print(f'{w_out.shape=}')
    print(f'{model.W_U[:, 1:].shape=}')

    state = einops.einsum(
        w_out,
        model.W_U[:, 1:],
        'n d, d m -> n m'
    )
    output_weights_in_logit_basis = t.zeros((num, 8 * 8), device=device)
    output_weights_in_logit_basis[:, stoi_indices] = state
    output_weights_in_logit_basis = einops.rearrange(output_weights_in_logit_basis, 'n (d1 d2) -> n d1 d2', d1=8, d2=8)

    plot_square_as_board(
        output_weights_in_logit_basis, 
        title=f"Output weights of top 10 neurons in layer 5, in the output logit basis",
        facet_col=0, 
        facet_labels=[f"L5N{n.item()}" for n in top_neurons]
    )

    # TOP MOVES:
    board_states = []
    for neuron in top_neurons:
        # Get max activating dataset aggregations
        neuron_acts = focus_cache["post", 5, "mlp"][:, :, neuron]
        top_moves = neuron_acts > neuron_acts.quantile(0.99)
        board_state_at_top_moves = focus_states_flipped_pm1[:, :-1][top_moves].float().mean(0)
        board_states.append(board_state_at_top_moves)
    board_states = t.stack(board_states)
    # neurons_focuses = einops.rearrange(focus, 'g m n -> n g m')[top_neurons]
    # top_moves = neurons_focuses > neurons_focuses.quantile(0.99, dim=0, keepdim=True)
    # print(f'{top_moves.shape=}')
    # nuerons_focus_states_flipped_value = einops.repeat(
    #     focus_states_flipped_value,
    #     'g m x y -> n g m x y',
    #     n=num
    # )
    # focus_states_flipped_pm1 = t.zeros_like( nuerons_focus_states_flipped_value, device=device)
    # focus_states_flipped_pm1[nuerons_focus_states_flipped_value==2] = -1.
    # focus_states_flipped_pm1[nuerons_focus_states_flipped_value==1] = 1.
    # print(f'{focus_states_flipped_pm1.shape=}')
    # board_states = focus_states_flipped_pm1[:, :, :-1][top_moves].float()
    # print(f'{board_states.shape=}')
    # board_states = einops.reduce(board_states, 'n g m x y -> n x y', 'mean')
    # print(f'{board_states.shape=}')
    plot_square_as_board(
        board_states, 
        title=f"Aggregated top 30 moves for each top 10 neuron in layer 5", 
        facet_col=0, 
        facet_labels=[f"L5N{n.item()}" for n in top_neurons]
    )
    '''
    The plots above for L5N566: output weights in output logit basis for this neuron is zero.
    E4 for max activating dataset aggregation is -1 while others are not (0..1).
    This activated when E4 is mine?
    '''

_investigate_more_neurons()
# %%
# Spectrum Plots
c0 = focus_states_flipped_pm1[:, :, 2, 0]
d1 = focus_states_flipped_pm1[:, :, 3, 1]
e2 = focus_states_flipped_pm1[:, :, 4, 2]

label = (c0==0) & (d1==-1) & (e2==1)

neuron =  1393
neuron_acts = focus_cache["post", 5][:, :, neuron]

def make_spectrum_plot(
    neuron_acts: Float[Tensor, "batch"],
    label: Bool[Tensor, "batch"],
    **kwargs
) -> None:
    '''
    Generates a spectrum plot from the neuron activations and a set of labels.
    '''
    px.histogram(
        pd.DataFrame({"acts": neuron_acts.tolist(), "label": label.tolist()}), 
        x="acts", color="label", histnorm="percent", barmode="group", nbins=100, 
        title=f"Spectrum plot for neuron L5N{neuron} testing C0==BLANK & D1==THEIRS & E2==MINE",
        color_discrete_sequence=px.colors.qualitative.Bold
    ).show()

make_spectrum_plot(neuron_acts.flatten(), label[:, :-1].flatten())
# %%
pp(f'Num of states with C0==BLANK & D1==THEIRS & E2==MINE: {label.int().sum()}')
pp(f'Num of games with C0==BLANK & D1==THEIRS & E2==MINE: {label.any(dim=1).int().sum()}')
low_labeled_moves_games = (label.float().sum(dim=1)).argsort()[-3:]

# %%
# for game in low_labeled_moves_games:
#     moves=slice(0,25)
#     imshow(
#         focus_states[game, moves],
#         facet_col=0,
#         facet_col_wrap=5,
#         y=list("ABCDEFGH"),
#         facet_labels=[f"Move {i}" for i in list(range(60))[moves]],
#         title=f"First moves of {game} game",
#         color_continuous_scale="Greys",
#         coloraxis_showscale=False,
#         width=1000,
#         height=1000,
#     )
# %%
game = 32
# Plot neuron_acts using plotly for the game 32 along each move
px.line(
    pd.DataFrame({"acts": neuron_acts[game].tolist()}),
    title=f"Neuron L5N{neuron} activations for game {game}",
    color_discrete_sequence=px.colors.qualitative.Bold,
).show()

moves=slice(30,49)
imshow(
    focus_states[game, moves],
    facet_col=0,
    facet_col_wrap=5,
    y=list("ABCDEFGH"),
    facet_labels=[f"Move {i}" for i in list(range(60))[moves]],
    title=f"Moves of {game} game",
    color_continuous_scale="Greys",
    coloraxis_showscale=False,
    width=1000,
    height=1000,
)
'''
Observations: (Wrong? because wrong plot above)
- In game 32, there are only 4 peaks in activation for the neuron 1393: 35, 37, 39, 42 board states. 
- 35 state has C0=BLANK & D1=THIERS & E2=MINE. Playing for black. 
  This is the _only_ way to play at C0 for blacks.
- 37 state is the same as 35 with regard to C0.
- 39 state. Has also _only one_ direction for black to 
  put its piece at C0 but now it is horizontal. 

'''
# %%
# Exercise - make more spectrum plots
layer = 5
neuron =  566
e5 = focus_states_flipped_pm1[:, :, 4, 5] 
label = e5 == -1  # E5 is mine.
neuron_acts = focus_cache["post", layer][:, :, neuron]

def make_spectrum_plot(
    neuron_acts: Float[Tensor, "batch"],
    label: Bool[Tensor, "batch"],
    **kwargs
) -> None:
    '''
    Generates a spectrum plot from the neuron activations and a set of labels.
    '''
    px.histogram(
        pd.DataFrame({"acts": neuron_acts.tolist(), "label": label.tolist()}), 
        x="acts", color="label", histnorm="percent", barmode="group", nbins=100, 
        title=f"Spectrum plot for neuron L5N{neuron} testing E5==MINE",
        color_discrete_sequence=px.colors.qualitative.Bold
    ).show()

make_spectrum_plot(neuron_acts.flatten(), label[:, :-1].flatten())
'''
Observations: the plot for the above has most of labeled and not labeled moves
in -0.2..0 interval (negative, GELU).
'''
# %%
labeled_moves_games = (label.float().sum(dim=1)).argsort()[-3:]
# %%
for game in labeled_moves_games:
    moves=slice(0,25)
    imshow(
        focus_states[game, moves],
        facet_col=0,
        facet_col_wrap=5,
        y=list("ABCDEFGH"),
        facet_labels=[f"Move {i}" for i in list(range(60))[moves]],
        title=f"First moves of {game} game",
        color_continuous_scale="Greys",
        coloraxis_showscale=False,
        width=1000,
        height=1000,
    )
# %%

# %%
game = 13
# Plot neuron_acts using plotly for the game 32 along each move
px.line(
    pd.DataFrame({"acts": neuron_acts[game].tolist()}),
    title=f"Neuron L5N{neuron} activations for game {game}",
    color_discrete_sequence=px.colors.qualitative.Bold,
).show()
'''
The plot above shows jagged line, with peaks at even (black) moves. Till move 56 (captured by white).
'''
# %%
game = 13
moves=slice(40,58)
imshow(
    focus_states[game, moves],
    facet_col=0,
    facet_col_wrap=5,
    y=list("ABCDEFGH"),
    facet_labels=[f"Move {i}" for i in list(range(60))[moves]],
    title=f"First moves of {game} game",
    color_continuous_scale="Greys",
    coloraxis_showscale=False,
    width=1000,
    height=1000,
)
# %%
