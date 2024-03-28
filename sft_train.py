import config
from utilities import (
    raw_dataset,
    sft_dataset__joey,
    initialize_model_and_tokenizer,
    find_all_linear_names,
    print_trainable_parameters,
)
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer


model, tokenizer = initialize_model_and_tokenizer()
model = prepare_model_for_kbit_training(
    model
)  # https://huggingface.co/docs/peft/v0.8.2/en/package_reference/peft_model#peft.prepare_model_for_kbit_training

# load Joey lines (with train/test split)
# We only use the dataframe index of manually select samples (`cp_list`) to train
dataset = sft_dataset__joey(raw_dataset(), tokenizer, test_size=0.1, cp_list=config.cp_list)


### TRAINING ###
train_args = TrainingArguments(
    # all params: https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    seed=config.SEED,
    output_dir=config.CHECKPOINTS_DIR,  # output dir where preds and model checkpoints will be saved
    #
    per_device_train_batch_size=2,  # use in coordination with gradient_accumulation_steps to maximize GPU memory usage
    gradient_accumulation_steps=1,  # compute gradients only after X batches/steps; enables larger batches than available GPU memory (with small increase in train time) https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-accumulation
    gradient_checkpointing=False,  # save a portion of forward activations (instead of all) for backward pass gradient calcs; use if memory constrained (note: training is ~20% longer).  # https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
    fp16=True,  # activations are calculated in fp16 precision; increases memory usage (model on GPU @ fp16 and fp32) but significantly speeds up computation
    # fsdp="full_shard",  # distributed training only; https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.fsdp
    group_by_length=True,  # group similar-lengthed sequences together
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
    report_to="tensorboard",
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
    target_modules=find_all_linear_names(
        model, check4bit=True, check8bit=False, verbose=True
    ),
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

if train_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

model = get_peft_model(
    model, lora_config
)  # https://huggingface.co/docs/peft/v0.8.2/en/package_reference/peft_model#peft.get_peft_model
model.print_trainable_parameters()

trainer = SFTTrainer(
    model=model,
    args=train_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    ),  # DCFLM dynamically pads items in each batch
    train_dataset=dataset["train"] if "train" in dataset else dataset,
    eval_dataset=dataset["test"] if "test" in dataset else None,
    # dataset_batch_size=1,
    peft_config=lora_config,
    max_seq_length=config.MAX_LENGTH,
    tokenizer=tokenizer,
    dataset_text_field="payload",
    packing=False,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# print(train_args)
# print(lora_config)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
print(metrics)


# Save model
trainer.model.peft_config["default"].base_model_name_or_path = config.MODEL_NAME
print("pushing fine-tuned model to HF...")
trainer.model.push_to_hub(config.SFT_MODEL_NAME)


print("pushing MERGED fine-tuned model to HF...")
merged_model = trainer.model.merge_and_unload()
merged_model.push_to_hub(config.SFT_MERGED_MODEL_NAME)