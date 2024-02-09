import config
from random import randrange

from utilities import raw_dataset, sft_dataset__joey, tokenize_batch

from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token  # llama tokenizer does not have a pad_token

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    quantization_config=config.BNB_CONFIG,
    device_map="auto",
    max_memory={i: f"{40960}MB" for i in range(config.N_GPUS)},
)

dataset = sft_dataset__joey(raw_dataset(), tokenizer, shuffle=True)


