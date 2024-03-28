import config
import utilities

import pandas as pd
from functools import partial

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig

from trl import RewardConfig, RewardTrainer

### LOAD REWARD MODEL AND TOKENIZER ###

model = AutoModelForSequenceClassification.from_pretrained(
    f"{config.USER}/{config.SFT_MERGED_MODEL_NAME}",
    num_labels=1,  # num_labels = num of categories
)
model.config.pad_token_id = model.config.eos_token_id

tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_NAME, padding_side="left"
)  # we use the original tokenizer for this
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = False


### PREP REWARD MODEL TRAIN DATASET ###

rt_preprocess_fuction_partial = partial(utilities.rt_preprocess_function, tokenizer)

df = pd.read_csv(config.REWARD_MODEL_TRAINING_DATASET_FILENAME)
df = utilities.rt_augment_dataset(df)
df.loc[:, "good"] = df.good.map(utilities.get_response)
df.loc[:, "bad"] = df.bad.map(utilities.get_response)

dataset = Dataset.from_pandas(df[["good", "bad"]])
dataset = dataset.map(
    rt_preprocess_fuction_partial,
    batched=True,
    num_proc=1,
)

dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)


### REWARD MODEL TRAINING ###

reward_config = RewardConfig(
    output_dir=f"{config.DIR}/{config.MODELS_DIR}/reward_model_test/",
    # max_length=400,
    remove_unused_columns=False,
    # all params: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    seed=42,
    #
    per_device_train_batch_size=2,  # use in coordination with gradient_accumulation_steps to maximize GPU memory usage
    gradient_accumulation_steps=1,  # compute gradients only after X batches/steps; enables larger batches than available GPU memory (with small increase in train time) https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-accumulation
    gradient_checkpointing=False,  # save a portion of forward activations (instead of all) for backward pass gradient calcs; use if memory constrained (note: training is ~20% longer).  # https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
    fp16=True,  # activations are calculated in fp16 precision; increases memory usage (model on GPU @ fp16 and fp32) but significantly speeds up computation
    # fsdp="full_shard",  # distributed training only; https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.fsdp
    group_by_length=False,  # RewardTrainer seems to break if True
    #
    num_train_epochs=1,
    logging_first_step=True,
    logging_strategy="steps",
    evaluation_strategy="steps",  # assess model perf after X number of steps
    save_strategy="steps",
    logging_steps=5,  # logging AND eval
    # save_steps=100,
    optim="paged_adamw_32bit",  # use "adafactor" for smaller memory footprint
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    # warmup_steps=int(len(dataset["train"]) * 0.05),
    # max_steps=len(dataset["train"]) * 2,
    report_to=None,
    save_safetensors=True,
    neftune_noise_alpha=0.1,
    max_grad_norm=0.5,  # max grad (gradient clipping)
    # group_by_length=True,
)


lora_config = LoraConfig(
    # https://huggingface.co/docs/peft/v0.8.2/en/package_reference/lora#peft.LoraConfig
    r=64,  # rank; number of trainable LoRA dims; max dim of adapter matrices (B and A): (d,r) * (r,k)
    lora_alpha=64,  # scaling factor for adapter weights where scaling = alpha/rank (if alpha>rank, higher impact of LoRA weights on base model weights)
    use_rslora=True,  # always. see https://arxiv.org/pdf/2312.03732.pdf
    init_lora_weights="gaussian",  # if using "loftq" do not pass a quantized model (loftQ qauantizes for you); also see `loftq_config`
    target_modules=["score"],  # only train the classifier layer/module
    # target_modules=find_all_linear_names(
    #     model, check4bit=True, check8bit=False, verbose=True
    # ),
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",  # text classificaiton; see other options here: https://discuss.huggingface.co/t/task-type-parameter-of-loraconfig/52879/6?u=braunagn
)

trainer = RewardTrainer(
    model=model,  # note: model is updated in place (no deep copy) with lora adapters.  run this step only once
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
)

trainer.train()

# push to hub
trainer.model.push_to_hub(f"{config.USER}/{config.REWARD_MODEL_NAME}")

# also save merged model to hub
merged_model = trainer.model.merge_and_unload()
merged_model.push_to_hub(f"{config.USER}/{config.REWARD_MERGED_MODEL_NAME}")
