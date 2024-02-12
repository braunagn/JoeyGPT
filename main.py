import config
from random import randrange

from utilities import raw_dataset, sft_dataset__joey, tokenize_batch

from transformers import AutoModelForCausalLM, AutoTokenizer

# import pandas as pd
import os

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token  # llama tokenizer does not have a pad_token


if not os.path.exists(config.SINGLE_MODEL_DIR):
    # for first iteration, create model dir, download, save
    os.mkdir(config.SINGLE_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=config.BNB_CONFIG,
        device_map="auto",
        # max_memory={i: f"{40960}MB" for i in range(config.N_GPUS)},
    )
    model.save_pretrained(config.SINGLE_MODEL_DIR)
else:
    AutoModelForCausalLM.from_pretrained(config.SINGLE_MODEL_DIR)

dataset = sft_dataset__joey(raw_dataset(), tokenizer, shuffle=True)


