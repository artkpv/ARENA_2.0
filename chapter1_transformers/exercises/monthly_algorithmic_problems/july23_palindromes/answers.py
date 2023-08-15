# %%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from pathlib import Path

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
    d_mlp=None # this is an attn-only model
)

state_dict = t.load(filename)

state_dict = model.center_writing_weights(t.load(filename))
state_dict = model.center_unembed(state_dict)
state_dict = model.fold_layer_norm(state_dict)
state_dict = model.fold_value_biases(state_dict)
model.load_state_dict(state_dict, strict=False);