# %%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from pathlib import Path
import einops
from pprint import pp
import circuitsvis as cv
from IPython.display import display

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "monthly_algorithmic_problems" / "july23_palindromes"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from monthly_algorithmic_problems.july23_palindromes.dataset import PalindromeDataset, display_seq
from monthly_algorithmic_problems.july23_palindromes.model import create_model
from plotly_utils import hist, bar, imshow

device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
filename = section_dir / "palindrome_classifier.pt"

model = create_model(
    half_length=10, # this is half the length of the palindrome sequences
    max_value=30, # values in palindrome sequence are between 0 and max_value inclusive
    seed=42,
    d_model=28,
    d_head=14,
    n_heads=2,
    normalization_type="LN",
    d_mlp=None, # this is an attn-only model
    device=device
)

state_dict = t.load(filename, map_location=device)

state_dict = model.center_writing_weights(t.load(filename, map_location=device))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);

# %%
print("ln_final weight: ", model.ln_final.w)
print("\nln_final, bias: ", model.ln_final.b)
W_U_mean_over_input = einops.reduce(model.W_U, "d_model d_vocab -> d_model", "mean")
t.testing.assert_close(W_U_mean_over_input, t.zeros_like(W_U_mean_over_input))

W_U_mean_over_output = einops.reduce(model.W_U, "d_model d_vocab -> d_vocab", "mean")
t.testing.assert_close(W_U_mean_over_output, t.zeros_like(W_U_mean_over_output))

W_O_mean_over_output = einops.reduce(model.W_O, "layer head d_head d_model -> layer head d_head", "mean")
t.testing.assert_close(W_O_mean_over_output, t.zeros_like(W_O_mean_over_output))

b_O_mean_over_output = einops.reduce(model.b_O, "layer d_model -> layer", "mean")
t.testing.assert_close(b_O_mean_over_output, t.zeros_like(b_O_mean_over_output))

W_E_mean_over_output = einops.reduce(model.W_E, "token d_model -> token", "mean")
t.testing.assert_close(W_E_mean_over_output, t.zeros_like(W_E_mean_over_output))

W_pos_mean_over_output = einops.reduce(model.W_pos, "position d_model -> position", "mean")
t.testing.assert_close(W_pos_mean_over_output, t.zeros_like(W_pos_mean_over_output))

b_V = model.b_V
t.testing.assert_close(b_V, t.zeros_like(b_V))
# %%
pp(model)
pp(f"{model.W_E.shape=}")
pp(f"{model.blocks[0].attn.W_Q.shape=}")
pp(f"{model.blocks[0].attn.W_V.shape=}")
pp(f"{model.blocks[0].attn.W_O.shape=}")
pp(f"{model.W_U.shape=}")


# %%
# %%
# Run 
dataset = PalindromeDataset(size=100, max_value=30, half_length=10)
ds_slice = slice(0,5)
toks, is_palindrome = dataset[ds_slice]

output, cache = model.run_with_cache(toks, return_type='logits')
logits = output[:, -1]
probs = logits.softmax(-1)
probs_palindrome = probs[:, 1]

# %% 
# Attention heads

layer = 1
example_id = 0
attn_patterns = cache['pattern', layer][example_id]
str_tokens = dataset.str_toks[ds_slice][example_id]
#pp(dataset.str_toks[ds_slice])
pp(attn_patterns.shape)
display(cv.attention.attention_patterns(
    str_tokens, # list of strings
    attention=attn_patterns # tensor of shape (n_heads, seq_len, seq_len),
))

# %%
# Logit attribution
# Calc which attn head contributes most to the final logit.

for tok, prob in zip(toks, probs_palindrome):
    display_seq(tok, prob)

pp(f'{toks.shape=}')
pp(f'{output.shape=}')
pp(f'{logits.shape=}')
pp(f'{cache=}')
pp(f'{cache["attn_out", 0].shape=}')
# %%
