import config

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
import torch
from tqdm import tqdm


### PREP DATASET ###

tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_NAME, padding_side="left"
)  # we use the original tokenizer for this
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = False

user_inputs = [
    "So, how was the date?",
    "Where are you going?",
    "Do you like her?",
    "How did the audition go?",
    "So I bumped into her again yesterday, what should I do?",
    "Hey Joey!",
    "When are we playing foosball next?",
    "Should I take the job?",
    "Where you going?",
    "What's up?",
    "Do you think she likes me or you more?",
    "Come on, tell me the truth.",
    "Where's your favorite pizza place?",
    "Come on, give me a slice of pizza!",
    "Joey, are you okay?",
    "What's the problem?",
    "What does your ideal date look like?",
    "Do you know her?",
    "Tell me when your next audition is!",
    "Whom do you live with?",
    "Where are you from?",
]

prompts = [
    {"query": config.TEMPLATE.format(prompt=i, response="")} for i in user_inputs
]
dataset = Dataset.from_list(prompts)


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


dataset = dataset.map(tokenize, batched=False)


### LOAD MODELS ###
sft_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    f"{config.USER}/{config.SFT_MERGED_MODEL_NAME}"
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    f"{config.USER}/{config.SFT_MERGED_MODEL_NAME}"
)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    f"{config.USER}/{config.REWARD_MERGED_MODEL_NAME}"
)


### SETUP PPO HYPERPARAMETERS ###

from trl import PPOConfig
from transformers import GenerationConfig


ppo_config = PPOConfig(
    # source: https://github.com/huggingface/trl/blob/v0.7.11/trl/trainer/ppo_config.py#L35
    # general
    exp_name=None,  # "experiment name; defaults to filename without extension"
    seed=config.SEED,
    log_with=None,
    task_name="PPO Training",  # used only for tracking purposes
    model_name=config.SFT_MERGED_MODEL_NAME,  # used only for tracking purposes
    query_dataset="custom",  # used only for tracking purposes
    reward_model=config.REWARD_MERGED_MODEL_NAME,  # used only for tracking purposes
    remove_unused_columns=True,
    # hyperparameters
    ppo_epochs=1,  # Number of optimization epochs per batch of samples
    steps=1,  # number of training steps
    batch_size=1,  # batch size per optimization step
    mini_batch_size=1,  # Number of samples optimized in each mini batch
    gradient_accumulation_steps=1,  # number of steps before loss calculation
    learning_rate=1e-4,
    adap_kl_ctrl=True,  # use adaptive KL control, otherwise linear
    kl_penalty="kl",  # kl penalty options: 'kl': model_logp - ref_logp,  'abs': abs(kl),  'mse': mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution
    target=6,  # Target KL value for adaptive KL control
    horizon=10000,  # Horizon for adaptive KL control
    gamma=0.99,  # Gamma parameter for advantage calculation
    lam=0.95,  # Lambda parameter for advantage calculation
    cliprange=0.2,  # Range for clipping in PPO policy gradient loss
    cliprange_value=0.2,  # Range for clipping values in loss calculation
    vf_coef=0.1,  # Scaling factor for value loss
    forward_batch_size=None,  # DEPRECATED, see mini_batch_size
    world_size=None,  # int, use for distributed training
    max_grad_norm=None,  # Maximum gradient norm for gradient clipping
    early_stopping=True,  # Whether to stop the PPO optimization loop early if the KL too high
    target_kl=1.0,  # Stop early if we exceed this value by over 50%
    compare_steps=1,  # Number of steps between comparison of the current reward with the best seen so far
    whiten_rewards=False,  # Whiten the rewards before compute advantages
)

generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    min_new_tokens=1,  # NEW tokens generated (doesn't include prompt)
    max_new_tokens=125,  # NEW tokens generated (doesn't include prompt)
    do_sample=True,  # sample across tokens; if false, model uses greedy decoding
    temperature=1.0,  # how randomly model samples from list of available tokens
    # top_p=0.30,  # list of tokens (top X %) which model can sample from
    # num_beams=1,  # beam search; number of beams (X more runs through model)
    penalty_alpha=0.6,  # contrastive search param alpha https://huggingface.co/blog/introducing-csearch#5-contrastive-search
    top_k=4,  # contrastive search param `k` https://huggingface.co/blog/introducing-csearch#5-contrastive-search
    # repetition_penalty=1.0,  # 1=no penalty; penalty at >1
    # length_penalty=1.0, # > 0.0 promotes longer sequences; < 0.0 encourages shorter sequences.
    # exponential_decay_length_penalty=(40, 1.1),  # increase likelihood of eos_token probability; increase starts at X generated tokens (X,_) and the penalty increases at Y (_,Y)
    num_return_sequences=1,  # how many times to perform independent generate from the prompt (default is 1)
    # eta_cutoff=8e-4,
    renormalize_logits=True,  # recommended by HF: https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig.renormalize_logits
    output_attentions=False,  # return attention tensors after generation
    output_hidden_states=False,  # return all hidden states of all layers
    output_scores=False,  # return the prediction scores
    return_dict_in_generate=False,  # return output as ModelOuput object instead of tuple
    use_cache=False,  # whether to re-use last passed key/value attentions
)

generation_kwargs = generation_config.to_dict()


### INITIALIZE PPO TRAINER ###
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=sft_model,
    ref_model=ref_model,  # will re-use `model` if set to None; hack= set to same as `model` to get around KeyError with quantized models
    tokenizer=tokenizer,
    dataset=dataset,
    optimizer=None,  # None defaults to Adam optimizer with linear learning rate specific in PPOConfig
    num_shared_layers=None,  # None defaults to all layers are shared  #####################
)

### TRAINING ###
for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.push_to_hub(f"{config.USER}/{config.PPO_MODEL_NAME}")
