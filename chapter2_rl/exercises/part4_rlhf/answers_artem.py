# %%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import json
import sys
import math
import gc
from pathlib import Path
import torch as t
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM,  AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertForMaskedLM
import logging
from typing import cast, Any, List, Optional, Union, Tuple
from re import search
from pprint import pprint

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_rlhf"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part4_rlhf.tests as tests
import part4_rlhf.utils as utils
from trlx.data.default_configs import TRLConfig, TrainConfig, OptimizerConfig, SchedulerConfig, TokenizerConfig, ModelConfig
from trlx.models.modeling_ppo import PPOConfig
from trlx import train

if t.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = -1
# %%
bert = utils.load_pretrained_bert()
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def predict(model: BertForMaskedLM, tokenizer: AutoTokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''

    # Make sure we're in eval mode
    model.eval()

    # Tokenizer returns a bunch of special BERT-specific things, we just want input ids
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    # Get top predictions at all the places we masked
    out = model(input_ids).logits
    preds = out[input_ids == tokenizer.mask_token_id]
    tops = preds.topk(k, dim=-1).indices

    return [[tokenizer.decode(t) for t in mask] for mask in tops]


your_text = "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK]."
predictions = predict(bert, tokenizer, your_text)
print("Model predicted: \n", "\n".join(map(str, predictions)))

# %%
your_text = "Mary and John walk into a store. Mary ask [MASK] to buy a yogurt."
predictions = predict(bert, tokenizer, your_text)
print("Model predicted: \n", "\n".join(map(str, predictions)))
# %%
for expression in (
    "2+2", "13+21", "42+85", "74+64", "19+73", "46-11", "1-15", "123+456", "753-125", "12*42", "31*41"
):
    ans = eval(expression)
    predictions = predict(bert, tokenizer, f"{expression} equals [MASK]")
    print(f"{expression=} {predictions=}")
    print("YES" if str(ans) in predictions else "NO")

# %%
imdb = load_dataset("imdb", split="train+test")
# %%
def label_split(dataset : Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]) -> Tuple[int, int]:
    neg = dataset['label'].count(0)
    pos = dataset['label'].count(1)

    #for text, label in dataset:
    #    if label == 0:
    #        neg += 1
    #    elif label  == 1:
    #        pos += 1
    return pos, neg

n_pos, n_neg = label_split(imdb)

tests.test_label_split(n_pos, n_neg)
# %%
def generate_prompts(dataset) -> List[str]:
    '''
    Generate & return prompts from dataset.
    We want to collect the first few (3-5, the choice is yours) words from each review to serve as prompts for our finetuned model. The generated text from these prompts will be later used to evaluate the performance of our finetuned model.
    '''
    return [
        search(r"^([^ ]+ ){3,7}", text).group()
        for text 
        in dataset['text']
        if text
    ]
    

prompts = generate_prompts(imdb)
pprint(prompts[:3])
# %%
def generate_completion(prompt, checkpoint=None) -> str:
    '''
    Loads the GPT2-IMDB tokenizer and model, and generates completions for the given prompt (in the form of a string).

    Find name of model & tokenizer at the documentation page: https://huggingface.co/lvwerra/gpt2-imdb.

    Remember to set the `do_sample=True` flag when you call `model.generate`.
    '''
    
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    model = AutoModelForCausalLM.from_pretrained(checkpoint or "lvwerra/gpt2-imdb")
    inputs = tokenizer(prompt, return_tensors='pt')
    tokens_out = model.generate(**inputs, do_sample=True, top_k=5, max_new_tokens=64).squeeze(0)
    outputs = tokenizer.decode(tokens_out)
    return outputs

res = generate_completion(prompts[53])
pprint(prompts[53])
pprint(res)

# %%
def reward_model(samples, **kwargs) -> List[float]:
    '''
    Returns the rewards for the given samples (according to model which is defined inside function body).

    kwargs are passed to your model during a forward pass.
    '''
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
    model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
    inputs = tokenizer(samples, padding=True, truncation=True, return_tensors="pt")
    with t.inference_mode():
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], **kwargs).logits

    res = t.softmax(logits, dim=-1)[:, 1].tolist()
    return res

# %%
example_strings = ["Example string", "I'm having a good day", "You are an ugly person"]
rewards = reward_model(example_strings)

tests.test_reward_model(rewards)
# %%
reward_model_batch_size = 128
def create_pipeline(model_path):

    return pipeline(
        "sentiment-analysis",
        model_path,
        top_k=2,
        truncation=True,
        batch_size=reward_model_batch_size,
        device=device,
    )

sentiment_fn = create_pipeline("lvwerra/distilbert-imdb")
sentiment_fn("I'm having a good day")
sentiment_fn([
    "I'm having a good day",
    "I'm having a bad day",
])
# %%
def reward_model(samples: List[str], **kwargs) -> List[float]:
    '''
    Returns a list of reward values corresponding to the samples in `samples`.
    '''
    return [s['score'] for out_ in sentiment_fn(samples) for s in out_ if s['label'] == 'POSITIVE']

reward_model(
    ["I'm having a good day", "I'm having a bad day"]
)
# %%
test_prompts = ['I am happy', 'I am sad']

rewards = reward_model(test_prompts)
tests.test_reward_test_prompts(rewards)

## Code below has an interesting set of examples:

print('I want to eat', reward_model('I want to eat'))
print('I want your puppy', reward_model('I want your puppy'))
print('I want to eat your puppy', reward_model('I want to eat your puppy'))
print("You're a sad bastard", reward_model("You're a sad bastard"))
print("You're a clever bastard", reward_model("You're a clever bastard"))
print("You're a savvy", reward_model("You're a savvy"))
print("You're a savvy puppy", reward_model("You're a puppy"))
print("You're a savvy bastard", reward_model("You're a savvy bastard"))
#print('Сегодня отличный день', reward_model('Сегодня отличный день'))  # Russian for "Today is a great day"
#print('Сегодня ужасный день', reward_model('Сегодня ужасный день'))
# %%
def ppo_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=10000,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
        ),
        model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=3e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=1e12, eta_min=3e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=4,
            init_kl_coef=0.001,
            target=None,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=64,
                top_k=10, # or can do top_p
                do_sample=True,
            ),
        ),
    )


def main() -> None:
    cfg = ppo_config()
    # Call the `train` function with appropriate arguments
    trainer = train(
        reward_fn = reward_model,
        prompts = prompts,
        eval_prompts =  ['I am quite interested '] * reward_model_batch_size,
        config = cfg
    )
    trainer.save_pretrained(Path('./chapter2_rl/exercises/part4_rlhf/data/ppo_trained_model'))


gc.collect()
t.cuda.empty_cache()
main()
# %%
generate_completion("I hate this movie!", checkpoint=Path('./chapter2_rl/exercises/part4_rlhf/data/ppo_trained_model'))
'''
Output:
'I hate this movie! And it is a great one! It is a great one for everyone! It is a great one! It is a great one! And it is a great one! I love it! And is a great one! It isiji is wonderful and I love it! It is great and is great! It is great'
'''
# %%
def main() -> None:
    return train(
        reward_fn = reward_model,
        prompts = prompts,
        eval_prompts = ['I was extremely disappointed'] * 52,
        config =  ppo_config()
    )

gc.collect()
t.cuda.empty_cache()
main()
# %%
def get_neutral_score(scores):
    d = dict(map(lambda x: tuple(x.values()), scores))
    return 1 - abs(d["POSITIVE"] - d["NEGATIVE"])

def neutral_reward_model(samples: List[str], **kwargs) -> List[float]:
    reward = list(map(get_neutral_score, sentiment_fn(samples)))
    return reward

def main() -> None:
    trainer = train(
        reward_fn = neutral_reward_model,
        prompts = prompts,
        eval_prompts = ['I did NOT like this movie.'] * 64,
        config =  ppo_config()
    )

gc.collect()
t.cuda.empty_cache()
main()
# %%
