#%%
import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, hist, plot_comp_scores, plot_logit_attribution, plot_loss_difference
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

from pprint import pp

# %%
if MAIN:
    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

    pp(gpt2_small.cfg)
    print(gpt2_small.cfg.n_layers)
    print(gpt2_small.cfg.n_heads)
    print(gpt2_small.cfg.n_ctx)


# %%
if MAIN:
    model_description_text = '''## Loading Models

    HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

    For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!'''

    loss = gpt2_small(model_description_text, return_type="loss")
    print("Model loss:", loss)

    logits = gpt2_small(model_description_text, return_type="logits")
    print(f"{logits.shape=}")
    tokens = gpt2_small.to_tokens(model_description_text)
    print(f"{tokens.shape=}")
# %%

if MAIN:
    print(f"{gpt2_small.W_E.shape=}")
    print(f"{gpt2_small.W_U.shape=}")
    print(f"{gpt2_small.W_pos.shape=}")

    print(f"{gpt2_small.W_in.shape=}")
    print(f"{gpt2_small.W_out.shape=}")

    print(f"{gpt2_small.W_K.shape=}")
    print(f"{gpt2_small.W_Q.shape=}")
    print(f"{gpt2_small.blocks[0].attn.W_Q.shape=}")
# %%
if MAIN:
    logits: Tensor = gpt2_small(model_description_text, return_type="logits")
    prediction = logits.argmax(dim=-1).squeeze()[:-1]
    tokens = gpt2_small.to_tokens(model_description_text)
    pp(f"{tokens.shape=}")
    pp(f"{prediction.shape=}")
    correct = tokens[0,1:] == prediction
    correct_num = (correct).sum()
    pp(correct_num)
    pp(correct_num / prediction.size(0))
    pp(gpt2_small.to_str_tokens(tokens[0,1:][correct]))

# %%
if MAIN:
    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
    gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)
# %%
if MAIN:
    #pp(gpt2_cache)
    attn_patterns_layer_0 = gpt2_cache["pattern", 0]
    attn_patterns_layer_0_copy = gpt2_cache["blocks.0.attn.hook_pattern"]
    t.testing.assert_close(attn_patterns_layer_0, attn_patterns_layer_0_copy)
    pp(attn_patterns_layer_0.shape)
    pp(attn_patterns_layer_0_copy.shape)
# %%
if MAIN:
    layer0_pattern_from_cache = gpt2_cache["pattern", 0]
    k = gpt2_cache["k", 0]
    q = gpt2_cache["q", 0]
    s, h, dh = k.shape
    attn_scores = einops.einsum(q, k, 'sq h dh, sk h dh -> h sq sk')
    mask = t.triu(t.ones_like(attn_scores), diagonal=1) == 1
    attn_scores.masked_fill_(mask, value=-1e9)
    attn_scores /= dh ** .5
    layer0_pattern_from_q_and_k = attn_scores.softmax(dim=-1)

    t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
    print("Tests passed!")

# %%
if MAIN:
    print(type(gpt2_cache))
    attention_pattern = gpt2_cache["pattern", 0, "attn"]
    print(attention_pattern.shape)
    gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

    print("Layer 0 Head Attention Patterns:")
    display(cv.attention.attention_patterns(
        tokens=gpt2_str_tokens, 
        attention=attention_pattern,
        #attention_head_names=[f"L0H{i}" for i in range(12)],   # Breaks for me.
    ))

# %%

########
# PART 2
########
# %%
if MAIN:
    cfg = HookedTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=50278,
        attention_dir="causal",
        attn_only=True, # defaults to False
        tokenizer_name="EleutherAI/gpt-neox-20b", 
        seed=398,
        use_attn_result=True,
        normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
        positional_embedding_type="shortformer"
    )

# %%
if MAIN:
    weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()

    if not weights_dir.exists():
        url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
        output = str(weights_dir)
        gdown.download(url, output)
# %%
if MAIN:
    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_dir, map_location=device)
    model.load_state_dict(pretrained_weights)
# %%
if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

    logits, cache = model.run_with_cache(text, remove_batch_dim=True)

    tokens = model.to_str_tokens(text)
    for l in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", l, "attn"]

        print(f"Layer {l} Head Attention Patterns:")
        display(cv.attention.attention_patterns(
            tokens=tokens, 
            attention=attention_pattern,
        ))

    # Notes:
    # - Skip-trigram. 'we think they would by default' in Layer 0. 'by' attends 'we think they would'
    # - Skip-trigram. L0 head 6: 'we' attends 'we' at the beginning of the sentence.
    # - Induction head at the second layer? 'machine' attends 'intelligence' in layer 1 head 10. 
# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    pass

def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    pass

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    pass


if MAIN:
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))