import config
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import time

s = time.time()

model = AutoModelForCausalLM.from_pretrained(f"{config.USER}/{config.SFT_MERGED_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True if "llama" in config.MODEL_NAME.lower() else False


generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    min_new_tokens=1,  # NEW tokens generated (doesn't include prompt)
    max_new_tokens=config.MAX_LENGTH,  # NEW tokens generated (doesn't include prompt)
    do_sample=True,  # sample across tokens; if false, model uses greedy decoding
    temperature=1.0,  # how randomly model samples from list of available tokens
    # top_p=0.30,  # list of tokens (top X %) which model can sample from
    # num_beams=1,  # beam search; number of beams (X more runs through model)
    penalty_alpha=0.6,  # contrastive search param alpha https://huggingface.co/blog/introducing-csearch#5-contrastive-search
    top_k=4,  # contrastive search param `k` https://huggingface.co/blog/introducing-csearch#5-contrastive-search
    # repetition_penalty=1.0,  # 1=no penalty; penalty at >1
    # length_penalty=1.0, # > 0.0 promotes longer sequences; < 0.0 encourages shorter sequences.
    # exponential_decay_length_penalty=(40, 1.1),  # increase likelihood of eos_token probability; increase starts at X generated tokens (X,_) and the penalty increases at Y (_,Y)
    num_return_sequences=2,  # how many times to perform independent generate from the prompt (default is 1)

    # eta_cutoff=8e-4,
    renormalize_logits=True,  # recommended by HF: https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig.renormalize_logits
    output_attentions=False,  # return attention tensors after generation
    output_hidden_states=False,  # return all hidden states of all layers
    output_scores=False,  # return the prediction scores
    return_dict_in_generate=False,  # return output as ModelOuput object instead of tuple
    use_cache=False,  # whether to re-use last passed key/value attentions
)


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
    "What's up",
    "Do you think she likes me or you more?",
    "Come on, tell me the truth.",
    "Where's your favorite pizza place?",
    "Come on, give me the pizza!",
    "Joey, are you okay?",
    "What's the problem?",
    "What does your ideal date look like?",
    "Do you know her?",
    "Tell me when your your next audition?",
    "Whom do you live with?",
    "Where are you from?",
]

prompts = [config.TEMPLATE.format(prompt=i, response="") for i in user_inputs]

encoding = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
tokenizer.batch_decode(encoding["input_ids"])


generated_ids = model.generate(**encoding, generation_config=generation_config)
out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

runtime = time.time() - s

df = pd.DataFrame(data={"out": out})
df.to_csv(f"{config.SINGLE_MODEL_DIR}/generated_responses__{runtime}.csv", index=False)

print(f"runtime: {runtime:0.1f} sec")
print("\n######################\n")
print("\n\n".join(out))