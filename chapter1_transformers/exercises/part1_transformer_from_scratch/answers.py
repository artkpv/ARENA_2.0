#%%
import os; os.environ['ACCELERATE_DISABLE_RICH'] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
from typing import Tuple, List, Optional, Dict
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

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
# import part1_transformer_from_scratch.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == '__main__'


if MAIN:
    reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
#%%
from pprint import pp

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

    pp(reference_gpt2.to_str_tokens(tokens))
# %%
batch = 1
position = 35
d_model = 768
n_heads = 12
n_layers = 12
d_mlp = 3072  # (4 * d_model)
d_head = 64 # (d_model / n_heads)

if MAIN:
    for activation_name, activation in cache.items():
        # Only print for first layer
        if ".0." in activation_name or "blocks" not in activation_name:
            print(f"{activation_name:30} {tuple(activation.shape)}")

#%%
if MAIN:
    for name, param in reference_gpt2.named_parameters():
        # Only print for first layer
        if ".0." in name or "blocks" not in name:
            print(f"{name:18} {tuple(param.shape)}")

# %%
# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures

if MAIN:
    print(reference_gpt2.cfg)

# %%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


if MAIN:
    cfg = Config()
    print(cfg)

# %%
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape, "\n")

def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple): output = output[0]
    print("Output shape:", output.shape)
    try: reference_output = gpt2_layer(input)
    except: reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")



# %%

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        m = t.mean(residual, dim=-1, keepdim=True)
        v = t.var(residual, dim=-1, unbiased=False, keepdim=True)
        res = (residual - m) / (v + self.cfg.layer_norm_eps).sqrt() * self.w + self.b
        return res

if MAIN:
    rand_float_test(LayerNorm, [2, 4, 768])
    load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
# %%

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        if self.cfg.debug:
            print(tokens.shape)
            print(self.W_E.shape)
        #return nn.functional.one_hot(tokens, num_classes=self.W_E.shape[0]).to(t.float) @ self.W_E
        return self.W_E[tokens]


if MAIN:
    rand_int_test(Embed, [2, 4])
    load_gpt2_test(Embed, reference_gpt2.embed, tokens)

# %%
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
            self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        if self.cfg.debug:
            print(f"{tokens.shape=}")
            print(f"{self.W_pos.shape=}")
        return einops.repeat(
            self.W_pos[:tokens.shape[1]],
            'n d -> b n d',
            b=tokens.shape[0]
        )

if MAIN:
    rand_int_test(PosEmbed, [2, 4])
    load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)

# import circuitsvis as cv
# from IPython.display import display
# 
# 
# if MAIN:
#     html = cv.attention.attention_patterns(
#         tokens=reference_gpt2.to_str_tokens(reference_text), 
#         attention=cache["pattern", 0][0]
#     )
#     display(html)
# 

#%%

class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        n_heads, d_model, d_head = self.W_Q.shape
        resid_broadcased = einops.repeat(normalized_resid_pre, 'b p dm -> b p h dm', h=n_heads)
        if self.cfg.debug:
            print(f'{resid_broadcased.shape=}')
            print(f'{self.W_Q.shape=}')
            print(f'{self.W_K.shape=}')

        # Step 1 
        query = einops.einsum(
            resid_broadcased, self.W_Q , 
            'b p h dm, h dm dh -> b p h dh'
        ) + self.b_Q
        key = einops.einsum(
            resid_broadcased,
            self.W_K,
            'b p h dm, h dm dh -> b p h dh'
        ) + self.b_K
        if self.cfg.debug:
            print(f'{query.shape=}', f'{key.shape=}')
        attn_scores = einops.einsum(
            query,
            key,
            'b sq h dh, b sk h dh -> b h sq sk'
        )
        attn_scores /= t.sqrt(t.tensor(d_head))
        attn_scores = self.apply_causal_mask(attn_scores)
        attn_scores = attn_scores.softmax(dim=(-1))
        if self.cfg.debug:
            print(f'{attn_scores.shape=}')

        # Step 2 
        value = einops.einsum(
            resid_broadcased,
            self.W_V,
            'b p h dm, h dm dh ->  b p h dh'
        ) + self.b_V
        if self.cfg.debug:
            print(f'{value.shape=}')
        z = einops.einsum(
            attn_scores,
            value,
            'b h sq p, b p h dh -> b h sq dh'
        )
        if self.cfg.debug:
            print(f'{z.shape=}')
        x = einops.einsum(
            z,
            self.W_O,
            'b h sq dh, h dh dm  -> b sq dm'
        ) + self.b_O
        if self.cfg.debug:
            print(f'{x.shape=}')
        return x


    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        mask = t.triu(
            t.ones(attn_scores.shape[-2:], device=attn_scores.device), 
            diagonal=1
        ).bool()
        return attn_scores.masked_fill_(mask, self.IGNORE)


if MAIN:
    rand_float_test(Attention, [2, 4, 768])
    load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])

# %%
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        x = einops.einsum(
            normalized_resid_mid,
            self.W_in,
            'b p dm, dm dp -> b p dp'
        ) + self.b_in
        x = gelu_new(x)
        x = einops.einsum(
            x,
            self.W_out,
            'b p dp, dp dm -> b p dm'
        ) + self.b_out
        return x

if MAIN:
    rand_float_test(MLP, [2, 4, 768])
    load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])

# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        x = resid_pre
        x = self.attn(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x

if MAIN:
    rand_float_test(TransformerBlock, [2, 4, 768])
    load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

# %%
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return einops.einsum(
            normalized_resid_final,
            self.W_U,
            'b p dm, dm dv -> b p dv'
        ) + self.b_U


if MAIN:
    rand_float_test(Unembed, [2, 4, 768])
    load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

# %%
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        x = self.embed(tokens) + self.pos_embed(tokens)
        for tb in self.blocks:
            x = tb(x)
        x = self.unembed(self.ln_final(x))
        return x


if MAIN:
    rand_int_test(DemoTransformer, [2, 4])
    load_gpt2_test(DemoTransformer, reference_gpt2, tokens)

# %%
def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], 
    tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens


if MAIN:
    demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
    demo_logits = demo_gpt2(tokens)

    pred_log_probs = get_log_probs(demo_logits, tokens)
    print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
    print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
    print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")
# %%
if MAIN:
    test_string = '''The Total Perspective Vortex derives its picture of the whole Universe on the principle of'''
    for i in tqdm(range(100)):
        test_tokens = reference_gpt2.to_tokens(test_string).to(device)
        demo_logits = demo_gpt2(test_tokens)
        test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

    print(test_string)

# %%
if MAIN:
    model_cfg = Config(
        debug=False, 
        d_model=256, 
        n_heads=4, 
        d_head=64, 
        d_mlp=1024, 
        n_layers=2, 
        n_ctx=256, 
        d_vocab=reference_gpt2.cfg.d_vocab
    )
    model = DemoTransformer(model_cfg)
# %%
@dataclass
class TransformerTrainingArgs():
    batch_size = 8
    max_epochs = 1
    max_steps = 1000
    log_every = 10
    lr = 1e-3
    weight_decay = 1e-2
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day1-transformer"
    run_name: Optional[str] = None
    log_every_n_steps: int = 1


if MAIN:
    args = TransformerTrainingArgs()

# %%
if MAIN:
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
    print(dataset)
    print(dataset[0]['text'][:200])
# %%
if MAIN:
    tokenized_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model.cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
    data_loader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
# %%
if MAIN:
    first_batch = data_loader.dataset[:args.batch_size]

    print(first_batch.keys())
    print(first_batch['tokens'].shape)
# %%
class LitTransformer(pl.LightningModule):
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer, data_loader: DataLoader):
        super().__init__()
        self.model = model
        self.cfg = model.cfg
        self.args = args
        self.data_loader = data_loader

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        logits = self.model(tokens)
        return logits

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Float[Tensor, ""]:
        '''
        Here you compute and return the training loss and some additional metrics for e.g. 
        the progress bar or logger.
        '''
        logits = self.model(batch['tokens']) 
        log_probs = get_log_probs(logits, batch['tokens'])
        loss = -log_probs.mean()
        self.log('Train_loss', loss)
        return loss

    def configure_optimizers(self):
        '''
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        '''
        return t.optim.AdamW(
            params=self.parameters(),
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay
        )

    def train_dataloader(self):
        return self.data_loader

# %%
if MAIN:
    litmodel = LitTransformer(args, model, data_loader)
    logger = WandbLogger(
        save_dir=args.log_dir,
        project=args.log_name,
        name=args.run_name,
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model=litmodel, train_dataloaders=litmodel.data_loader)
    wandb.finish()
# %%
if MAIN:
    toks = tokenized_dataset[:]["tokens"].flatten()

    d_vocab = model.cfg.d_vocab
    freqs = t.bincount(toks, minlength=d_vocab)
    probs = freqs.float() / freqs.sum()

    distn = t.distributions.categorical.Categorical(probs=probs)
    entropy = distn.entropy()

    print(f"Entropy of training data = {entropy}")
# %%
