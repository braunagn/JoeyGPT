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
    #     AutoPeftModelForCausalLM,
)
# from trl import SFTTrainer


model, tokenizer = initialize_model_and_tokenizer()
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
target_modules = find_all_linear_names(
    model, check4bit=True, check8bit=False, verbose=True
)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

# load Joey lines
dataset = sft_dataset__joey(raw_dataset(), tokenizer, shuffle=True)
print(dataset[0])
# TRAINING
train_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    fp16=True,
    seed=42,
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
    # max_grad_norm=0.3,
    # group_by_length=True,
)

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=train_args,
    # peft_config=lora_config,
    # max_seq_length=1024,
    # packing=True,
    # tokenizer=tokenizer,
    # dataset_text_field="input",
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
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
