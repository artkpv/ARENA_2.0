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
    A = t.randn(5, 2)
    B = t.randn(2, 5)
    AB = A @ B
    AB_factor = FactoredMatrix(A, B)
    print("Norms:")
    print(AB.norm())
    print(AB_factor.norm())

    print(f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}")

# %%
if MAIN:
    print("Eigenvalues:")
    print(t.linalg.eig(AB).eigenvalues)
    print(AB_factor.eigenvalues)
    print()
    print("Singular Values:")
    print(t.linalg.svd(AB).S)
    print(AB_factor.S)
    print("Full SVD:")
    print(AB_factor.svd())
# %%
if MAIN:
    C = t.randn(5, 300)
    ABC = AB @ C
    ABC_factor = AB_factor @ C
    print("Unfactored:", ABC.shape, ABC.norm())
    print("Factored:", ABC_factor.shape, ABC_factor.norm())
    print(f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")
# %%
if MAIN:
    AB_unfactored = AB_factor.AB
    t.testing.assert_close(AB_unfactored, AB)

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

if MAIN:
    weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()
    if not weights_dir.exists():
        url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
        output = str(weights_dir)
        gdown.download(url, output)

if MAIN:
    model = HookedTransformer(cfg)
    pretrained_weights = t.load(weights_dir, map_location=device)
    model.load_state_dict(pretrained_weights)

# %%
if MAIN:
    pp(model.W_E.shape)
    pp(model.W_O.shape)
    pp(model.W_V.shape)
    pp(model.W_U.shape)
    l = 1
    h = 4
    #for l in range(model.cfg.n_layers):
    #    for h in range(model.cfg.n_heads):
    full_OV_circuit = model.W_E @ FactoredMatrix(model.W_V[l,h], model.W_O[l,h]) @ model.W_U
    tests.test_full_OV_circuit(full_OV_circuit, model, l, h)

# %%
if MAIN:
    indices = t.randint(0, model.cfg.d_vocab, (200,))
    full_OV_circuit_sample = full_OV_circuit[indices, indices].AB
    imshow(
        full_OV_circuit_sample,
        labels={"x": "Input token", "y": "Logits on output token"},
        title="Full OV circuit for copying head",
        width=700,
    )

# %%
def top_1_acc(full_OV_circuit: FactoredMatrix, batch_size: int = 100) -> float: 
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    A, B = full_OV_circuit.A, full_OV_circuit.B
    nrows = full_OV_circuit.shape[0]
    nrows_max_on_diagonal = 0

    for i in tqdm(range(0, nrows + batch_size, batch_size)):
        rng = range(i, min(i + batch_size, nrows))
        if rng:
            submatrix = A[rng, :] @ B
            diag_indices = t.tensor(rng, device=submatrix.device)
            nrows_max_on_diagonal += (submatrix.argmax(-1) == diag_indices).float().sum().item()

    return nrows_max_on_diagonal / nrows

if MAIN:
    print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(full_OV_circuit):.4f}")

# %%
if MAIN:
    pp(model.W_E.shape)
    pp(model.W_O.shape)
    pp(model.W_V.shape)
    pp(model.W_U.shape)
    layer = 1
    WV = einops.rearrange(model.W_V[1, (4,10)], 'h dm dh -> dm (h dh)')
    WO = einops.rearrange(model.W_O[1, (4,10)], 'h dh dm -> (h dh) dm')
    full_OV_circuit = model.W_E @ FactoredMatrix(WV, WO) @ model.W_U
    print(f"Fraction of the time that the best logit is on the diagonal: {top_1_acc(full_OV_circuit):.4f}")

# %%
def mask_scores(attn_scores: Float[Tensor, "query_nctx key_nctx"]):
    '''Mask the attention scores so that tokens don't attend to previous tokens.'''
    assert attn_scores.shape == (model.cfg.n_ctx, model.cfg.n_ctx)
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores


if MAIN:
    # YOUR CODE HERE - calculate the matrix `pos_by_pos_pattern` as described above
    layer = 0
    head_index = 7
    pp(model.W_Q.shape)
    pp(model.W_K.shape)
    pos_by_pos_pattern = (
        model.W_pos 
        @ model.W_Q[layer, head_index] 
        @ model.W_K[layer, head_index].T 
        @ model.W_pos.T
    )
    scores = mask_scores(pos_by_pos_pattern / model.cfg.d_head ** .5 )
    prob = scores.softmax(dim=-1)
    tests.test_pos_by_pos_pattern(prob, model, layer, head_index)
# %%
if MAIN:
    print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")

    imshow(
        utils.to_numpy(pos_by_pos_pattern[:100, :100]), 
        labels={"x": "Key", "y": "Query"}, 
        title="Attention patterns for prev-token QK circuit, first 100 indices",
        width=700
    )

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

# %%
def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, seq, d_model]

    The [i, :, :]th element is y_i (from notation above)
    '''
    return t.concat([
        cache["embed"].unsqueeze(0),
        cache["pos_embed"].unsqueeze(0),
        einops.rearrange(cache["result", 0], 's h dm -> h s dm')
    ], dim=0)

def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values)
    '''
    return einops.einsum(
        decomposed_qk_input,
        model.W_Q[1, ind_head_index],
        'i p dh, dh dm -> i p dm'
    )

def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_K (so the sum along axis 0 is just the k-values)
    '''
    return einops.einsum(
        decomposed_qk_input,
        model.W_K[1 ,ind_head_index],
        'i p dh, dh dm -> i p dm'
    )

if MAIN:
    ind_head_index = 4
    # First we get decomposed q and k input, and check they're what we expect
    decomposed_qk_input = decompose_qk_input(rep_cache)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1] + rep_cache["pos_embed"], rtol=0.01, atol=1e-05)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01)
    # Second, we plot our results
    component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
    for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
        imshow(
            utils.to_numpy(decomposed_input.pow(2).sum([-1])), 
            labels={"x": "Position", "y": "Component"},
            title=f"Norms of components of {name}", 
            y=component_labels,
            width=1000, height=400
        )
# %%
def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    '''
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, :, :]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    '''
    return einops.einsum(
        decomposed_q,
        decomposed_k,
        'i qp dm, j kp dm -> i j qp kp'
    )

if MAIN:
    tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k)

# %%
if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = einops.reduce(
        decomposed_scores, 
        "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", 
        t.std
    )

    # First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
    imshow(
        utils.to_numpy(t.tril(decomposed_scores[0, 9])), 
        title="Attention score contributions from (query, key) = (embed, output of L0H7)",
        width=800
    )

    # Second plot: std dev over query and key positions, shown by component
    imshow(
        utils.to_numpy(decomposed_stds), 
        labels={"x": "Key Component", "y": "Query Component"},
        title="Standard deviations of attention score contributions (by key and query component)", 
        x=component_labels, 
        y=component_labels,
        width=800
    )

# %%
def find_K_comp_full_circuit(
    model: HookedTransformer,
    prev_token_head_index: int,
    ind_head_index: int
) -> FactoredMatrix:
    '''
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    query = FactoredMatrix(
        model.W_E,
        model.W_Q[1, ind_head_index]
    )
    key = (
        model.W_E 
        @ FactoredMatrix(
            model.W_V[0,prev_token_head_index],
            model.W_O[0,prev_token_head_index]
        ) @ model.W_K[1, ind_head_index]
    )
    return query @ key.T


if MAIN:
    prev_token_head_index = 7
    ind_head_index = 4
    K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)

    tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

    print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc(K_comp_circuit.T):.4f}")
    # > 0.5201

#%%
if MAIN:
    prev_token_head_index = 7
    ind_head_index = 10
    K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)

    tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

    print(f"Fraction of tokens where the highest activating key is the same token: {top_1_acc(K_comp_circuit.T):.4f}")
    # > 0.5520

#%%

###########################################
# Further Exploration of Induction Circuits
###########################################
# %%
def get_comp_score(
    W_A: Float[Tensor, "in_A out_A"], 
    W_B: Float[Tensor, "out_A out_B"]
) -> float:
    '''
    Return the composition score between W_A and W_B.
    '''
    return (t.linalg.norm(W_A @ W_B) / (t.linalg.norm(W_A) * t.linalg.norm(W_B))).item()


if MAIN:
    tests.test_get_comp_score(get_comp_score)
    print('yes')
# %%
if MAIN:
    # Get all QK and OV matrices
    W_QK = model.W_Q @ model.W_K.transpose(-1, -2)
    W_OV = model.W_V @ model.W_O

    # Define tensors to hold the composition scores
    composition_scores = {
        "Q": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
        "K": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
        "V": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
    }

    # YOUR CODE HERE - fill in each tensor in the dictionary, by looping over W_A and W_B from layers 0 and 1
    for i in tqdm(range(model.cfg.n_heads)):
        for j in range(model.cfg.n_heads):
            composition_scores["Q"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j])
            composition_scores["K"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j].T)
            composition_scores["V"][i, j] = get_comp_score(W_OV[0, i], W_OV[1, j])

    for comp_type in "QKV":
        plot_comp_scores(model, composition_scores[comp_type], f"{comp_type} Composition Scores").show()
# %%

def generate_single_random_comp_score() -> float:
    '''
    Write a function which generates a single composition score for random matrices
    '''
    W_A_left = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_left = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_A_right = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_right = t.empty(model.cfg.d_model, model.cfg.d_head)

    for W in [W_A_left, W_B_left, W_A_right, W_B_right]:
        nn.init.kaiming_uniform_(W, a=np.sqrt(5))

    W_A = W_A_left @ W_A_right.T
    W_B = W_B_left @ W_B_right.T

    return get_comp_score(W_A, W_B)

if MAIN:
    n_samples = 300
    comp_scores_baseline = np.zeros(n_samples)
    for i in tqdm(range(n_samples)):
        comp_scores_baseline[i] = generate_single_random_comp_score()
    print("\nMean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    hist(
        comp_scores_baseline, 
        nbins=50, 
        width=800, 
        labels={"x": "Composition score"}, 
        title="Random composition scores"
    )

    baseline = comp_scores_baseline.mean()
    for comp_type, comp_scores in composition_scores.items():
        plot_comp_scores(model, comp_scores, f"{comp_type} Composition Scores", baseline=baseline).show()
# %%
def get_batched_comp_scores(
    W_As: FactoredMatrix,
    W_Bs: FactoredMatrix
) -> t.Tensor:
    '''Computes the compositional scores from indexed factored matrices W_As and W_Bs.

    Each of W_As and W_Bs is a FactoredMatrix object which is indexed by all but its last 2 dimensions, i.e.:
        W_As.shape == (*A_idx, A_in, A_out)
        W_Bs.shape == (*B_idx, B_in, B_out)
        A_out == B_in

    Return: tensor of shape (*A_idx, *B_idx) where the [*a_idx, *b_idx]th element is the compositional score from W_As[*a_idx] to W_Bs[*b_idx].
    '''
    W_As = FactoredMatrix(
        W_As.A.reshape(-1, 1, *W_As.A.shape[-2:]),
        W_As.B.reshape(-1, 1, *W_As.B.shape[-2:]),
    )
    W_Bs = FactoredMatrix(
        W_Bs.A.reshape(1, -1, *W_Bs.A.shape[-2:]),
        W_Bs.B.reshape(1, -1, *W_Bs.B.shape[-2:]),
    )

    # Compute the product
    W_ABs = W_As @ W_Bs

    # Compute the norms, and return the metric
    return W_ABs.norm() / (W_As.norm() * W_Bs.norm())

if MAIN:
    W_QK = FactoredMatrix(model.W_Q, model.W_K.transpose(-1, -2))
    W_OV = FactoredMatrix(model.W_V, model.W_O)

    composition_scores_batched = dict()
    composition_scores_batched["Q"] = get_batched_comp_scores(W_OV[0], W_QK[1])
    composition_scores_batched["K"] = get_batched_comp_scores(W_OV[0], W_QK[1].T) # Factored matrix: .T is interpreted as transpose of the last two axes
    composition_scores_batched["V"] = get_batched_comp_scores(W_OV[0], W_OV[1])

    t.testing.assert_close(composition_scores_batched["Q"], composition_scores["Q"])
    t.testing.assert_close(composition_scores_batched["K"], composition_scores["K"])
    t.testing.assert_close(composition_scores_batched["V"], composition_scores["V"])
    print("Tests passed - your `get_batched_comp_scores` function is working!")

# %%
def ablation_induction_score(prev_head_index: Optional[int], ind_head_index: int) -> float:
    '''
    Takes as input the index of the L0 head and the index of the L1 head, and then runs with the previous token head ablated and returns the induction score for the ind_head_index now.
    '''

    def ablation_hook(v, hook):
        if prev_head_index is not None:
            v[:, :, prev_head_index] = 0.0
        return v

    def induction_pattern_hook(attn, hook):
        hook.ctx[prev_head_index] = attn[0, ind_head_index].diag(-(seq_len - 1)).mean()

    model.run_with_hooks(
        rep_tokens,
        fwd_hooks=[
            (utils.get_act_name("v", 0), ablation_hook),
            (utils.get_act_name("pattern", 1), induction_pattern_hook)
        ],
    )
    return model.blocks[1].attn.hook_pattern.ctx[prev_head_index].item()

if MAIN:
    baseline_induction_score = ablation_induction_score(None, 4)
    print(f"Induction score for no ablations: {baseline_induction_score:.5f}\n")
    for i in range(model.cfg.n_heads):
        new_induction_score = ablation_induction_score(i, 4)
        induction_score_change = new_induction_score - baseline_induction_score
        print(f"Ablation score change for head {i:02}: {induction_score_change:+.5f}")
# %%
