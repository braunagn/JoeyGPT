import config
from utilities import (
    raw_dataset,
    sft_dataset__joey,
    tokenize_batch,
    initialize_model_and_tokenizer,
    find_all_linear_names,
    print_trainable_parameters,
)

from random import randrange

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    # PeftModel,
    # PeftConfig,
    # AutoPeftModelForCausalLM,
)
from trl import SFTTrainer

# tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
# tokenizer.pad_token = tokenizer.eos_token  # llama tokenizer does not have a pad_token
# tokenizer.add_eos_token = True

model, tokenizer = initialize_model_and_tokenizer()
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
target_modules = find_all_linear_names(
    model, check4bit=True, check8bit=False, verbose=True
)

# load Joey lines (with train/test split)
dataset = sft_dataset__joey(raw_dataset(), tokenizer, test_size=0.2)


### TRAINING ###

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

train_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    fp16=True,
    seed=config.SEED,
    output_dir=config.CHECKPOINTS_DIR,
    save_strategy="epoch",
    report_to="tensorboard",
    save_safetensors=True,
    optim="paged_adamw_32bit",
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=int(len(dataset) * 0.05),
    max_steps=len(dataset) * 1,
    logging_steps=50,
    neftune_noise_alpha=0.1,
    # max_grad_norm=0.3,
    # group_by_length=True,
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

trainer = SFTTrainer(
    model=model,
    args=train_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),  #DCFLM dynamically pads items in each batch
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    max_seq_length=config.MAX_LENGTH,
    tokenizer=tokenizer,
    dataset_text_field="payload",
    packing=False,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
print(metrics)


# # Save model
# print("Saving last checkpoint of the model...")
# os.makedirs(OUTPUT_DIR, exist_ok = True)
# trainer.model.save_pretrained(OUTPUT_DIR)
