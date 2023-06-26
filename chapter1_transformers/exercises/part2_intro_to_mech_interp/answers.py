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
        attention_pattern = cache[f"blocks.{l}.attn.hook_pattern"]

        print(f"Layer {l} Head Attention Patterns:")
        display(cv.attention.attention_patterns(
            tokens=tokens, 
            attention=attention_pattern,
        ))

    '''
    Examples:
    1. Layer 1, head 10. Skip-trigram attention. 'we' attends 'think' at the begining of the sentence, copying it to the output.
    2. Induction head at the second layer? 
        - 'machine' attends 'intelligence' in layer 1 head 10. 
        - 'this' at the end attends 'ectury' and 'level' which are preceeded by 'this'.
           - Head 10.
        - 'are' attends 'deceptive' and 'scaled' (preceeded by 'were' though) - Head 10.
        - '.' at the end attends ' If' - Head 8.
    3. Bigram? 'manip' and 'pulative' Layer 0, head 9.
    '''
# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    '''
    global model
    res = []
    for l in range(model.cfg.n_layers):
        heads = cache[f"blocks.{l}.attn.hook_pattern"]
        h_num, d_head, _ = heads.shape
        for hi in range(h_num):
            k = f'{l}.{hi}'
            diff_ = (heads[hi].argmax(dim=-1) - t.arange(0, d_head)).abs()
            mean_ = diff_.float().mean()
            if mean_ < 0.8:
                res += [k]
    return res


def prev_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    res = []
    for l in range(model.cfg.n_layers):
        heads = cache[f"blocks.{l}.attn.hook_pattern"]
        h_num, d_head, _ = heads.shape
        for hi in range(h_num):
            k = f'{l}.{hi}'
            diff_ = (heads[hi].argmax(dim=-1) - t.arange(0, d_head)).abs()
            mean_ = diff_.float().mean()
            if 0.8 <= mean_ < 1.8:
                res += [k]
    return res

def first_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    '''
    global model
    res = []
    for l in range(model.cfg.n_layers):
        heads = cache[f"blocks.{l}.attn.hook_pattern"]
        for hi in range(heads.shape[0]):
            max_ = heads[hi].argmax(dim=-1).float()
            mean_ = max_.mean().item()
            if mean_ < 1.0:
                res += [f'{l}.{hi}']
    return res

#%%
if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    #logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    pp(f"{logits.shape=}")
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
# %%

if MAIN:
    pass  # Uses too much memory.
    text = '''Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn’t think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley’s sister, but they hadn’t met for several years; in fact, Mrs. Dursley pretended she didn’t have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn’t want Dudley mixing with a child like that.'''

    logits, cache = model.run_with_cache(text, remove_batch_dim=True)

    tokens = model.to_str_tokens(text)
    for l in range(model.cfg.n_layers):
        attention_pattern = cache[f"blocks.{l}.attn.hook_pattern"]

        print(f"Layer {l} Head Attention Patterns:")
        display(cv.attention.attention_patterns(
            tokens=tokens, 
            attention=attention_pattern,
        ))

    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    # Output:
    # Heads attending to first token    =  0.3
    # Heads attending to current token  =  0.9
    # Heads attending to previous token =  0.7, 0.11
# %%

def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    res = t.randint(0, model.cfg.d_vocab, (batch, 1+2*seq_len))
    res[:,0] = prefix[:,0]
    res[:,1+seq_len:] = res[:,1:1+seq_len]
    return res

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return (rep_tokens, rep_logits, rep_cache)


if MAIN:
    seq_len = 50
    batch = 1
    (rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_cache.remove_batch_dim()
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

    print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

    plot_loss_difference(log_probs, rep_str, seq_len)

#%%
if MAIN:
    for l in range(model.cfg.n_layers):
        attention_pattern = rep_cache[f"blocks.{l}.attn.hook_pattern"]

        print(f"Layer {l} Head Attention Patterns:")
        display(cv.attention.attention_patterns(
            tokens=rep_str, 
            attention=attention_pattern,
        ))

# %%

def head_induction_attn_detector(head: t.Tensor) -> bool:
    '''
    Calculates the induction score for a single head.

    Inputs:
        head: [d_head, d_head] attention pattern for a single head

    Outputs:
        True or False, whether the head is an induction head, or not.
    '''
    d_head, _ = head.shape
    max_ = head.argmax(dim=-1)
    non_zero = max_ != 0
    for start in range(2, d_head-d_head//7):
        diagonal = t.cat([
            t.zeros(start),
            t.arange(0, d_head-start)
        ]).int().to(device)
        assert max_.shape == diagonal.shape
        diff_ = (max_[non_zero] - diagonal[non_zero]).abs()
        mean_ = diff_.float().mean()
        if mean_ < 0.4:
            return True
    return False

def induction_attn_detector(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    '''
    global model
    res = []
    heads = cache["blocks.1.attn.hook_pattern"]
    h_num, d_head, _ = heads.shape
    for hi in range(h_num):
        k = f'1.{hi}'
        if head_induction_attn_detector(heads[hi]):
            res += [k]
    return res


if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
# %%

###############
# PART 3. HOOKS
###############

# %%
if MAIN:
    seq_len = 50
    batch = 10
    rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

    # We make a tensor to store the induction score for each head.
    # We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    induction_score_store = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    '''
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    '''
    global induction_score_store
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    induction_score_store[hook.layer(), :] = induction_score

    # Slow:
    # batches, heads, ps, _ = pattern.shape
    # layer = hook.layer()
    # if layer == 0:
    #     return
    # for h in range(heads):
    #     score = 0.0
    #     for b in range(batches):
    #         head_pattern = pattern[b,h]
    #         if head_induction_attn_detector(head_pattern):
    #             score += 1.0
    #     score = score / (batches)
    #     induction_score_store[layer, h] = score

# %%

if MAIN:
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    model.run_with_hooks(
        rep_tokens_10, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            pattern_hook_names_filter,
            induction_score_hook
        )]
    )

    # Plot the induction scores for each head in each layer
    imshow(
        induction_score_store, 
        labels={"x": "Head", "y": "Layer"}, 
        title="Induction Score by Head", 
        text_auto=".2f",
        width=900, height=400
    )

# %%
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), 
            attention=pattern.mean(0)
        )
    )


if MAIN:
    seq_len = 50
    batch = 10
    rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

    # We make a tensor to store the induction score for each head.
    # We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
    induction_score_store = t.zeros((gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=model.cfg.device)

    pattern_hook_names_filter = lambda name: name.endswith("pattern")
    # Run with hooks (this is where we write to the `induction_score_store` tensor`)
    gpt2_small.run_with_hooks(
        rep_tokens_10, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            pattern_hook_names_filter,
            induction_score_hook
        )]
    )

    # Plot the induction scores for each head in each layer
    imshow(
        induction_score_store, 
        labels={"x": "Head", "y": "Layer"}, 
        title="Induction Score by Head", 
        text_auto=".2f",
        width=900, height=400
    )

    for induction_head_layer in [5, 6, 7]:
        gpt2_small.run_with_hooks(
            rep_tokens, 
            return_type=None, # For efficiency, we don't need to calculate the logits
            fwd_hooks=[
                (utils.get_act_name("pattern", induction_head_layer), visualize_pattern_hook)
            ]
        )

# %%
#################################
# Building interpretability tools
#################################
# %%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"]
) -> Float[Tensor, "seq-1 n_components"]:
    '''
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    '''
    W_U_correct_tokens = W_U[:, tokens[1:]]
    dir_attribution = einops.einsum(embed[:-1], W_U_correct_tokens, "s dm, dm s -> s").unsqueeze(-1)
    layer_0_attribution = einops.einsum(l1_results[:-1], W_U_correct_tokens, "s nh dm, dm s-> s nh")
    layer_1_attribution = einops.einsum(l2_results[:-1], W_U_correct_tokens, "s nh dm, dm s -> s nh")
    return t.concat([
        dir_attribution,
        layer_0_attribution,
        layer_1_attribution
    ], dim=-1)


if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    logits, cache = model.run_with_cache(text, remove_batch_dim=True)
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)

    with t.inference_mode():
        embed = cache["embed"]
        l1_results = cache["result", 0]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
        print("Tests passed!")
# %%
if MAIN:
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])

    plot_logit_attribution(model, logit_attr, tokens)
# %%
if MAIN:
    seq_len = 50

    embed = rep_cache["embed"]
    l1_results = rep_cache["result", 0]
    l2_results = rep_cache["result", 1]
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]

    # YOUR CODE HERE - define `first_half_logit_attr` and `second_half_logit_attr`
    first_half_logit_attr = logit_attribution(
        embed[:seq_len+1], 
        l1_results[:seq_len+1], 
        l2_results[:seq_len+1], 
        model.W_U, 
        first_half_tokens
    )
    second_half_logit_attr = logit_attribution(
        embed[seq_len:], 
        l1_results[seq_len:], 
        l2_results[seq_len:], 
        model.W_U, 
        second_half_tokens
    )
    assert first_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)
    assert second_half_logit_attr.shape == (seq_len, 2*model.cfg.n_heads + 1)

    plot_logit_attribution(model, first_half_logit_attr, first_half_tokens, "Logit attribution (first half of repeated sequence)")
    plot_logit_attribution(model, second_half_logit_attr, second_half_tokens, "Logit attribution (second half of repeated sequence)")
# %%

###################################
# Hooks: Intervening on Activations
###################################
# %%
def head_ablation_hook(
    v: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_head"]:
    v[:,:,head_index_to_ablate,:].fill_(0.0)
    return v


def cross_entropy_loss(logits, tokens):
    '''
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    '''
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def get_ablation_scores(
    model: HookedTransformer, 
    tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    '''
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("v", layer), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores



if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)
# %%
if MAIN:
    imshow(
        ablation_scores, 
        labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
        title="Logit Difference After Ablating Heads", 
        text_auto=".2f",
        width=900, height=400
    )
# %%
