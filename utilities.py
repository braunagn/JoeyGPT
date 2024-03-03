import config
from pathlib import Path
import os
import pandas as pd
import re
from datasets import Dataset
from functools import partial

from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb


def cleanup_speakers(line):
    # see SPEAKER_REPLACEMENTS for cleanup instructions
    replaced = False
    for sr in config.SPEAKER_REPLACEMENTS:
        for r in sr["replace"]:
            if r in line:
                return line.replace(r, sr["with"])
    if replaced is False:
        return line


def check_if_speaker_is(line, char):
    # True if char in <speaker:>, else False
    if char.lower() in line[: line.find(":")].lower():
        return True
    return False


def speaker_is(line):
    return line[: line.index(":")]


def remove_writer_notes(line):
    if "{" in line and "}" in line:
        start = line.index("{")
        end = line.index("}")
        return line[:start] + line[end + 1 :]
    return line


def get_spoken_lines(line):
    return line[line.index(":") + 1 :].strip()


def remove_extra_spaces(line):
    if "    " not in line and "   " not in line:
        return line
    else:
        if "    " in line:
            return line.replace("    ", " ")  # len=4
        return line.replace("   ", " ")  # len=3
    raise ValueError(f"line does not meet any criteria:\n {line}")


def compress_lines(lines, char, fuzzy=False, verbose=True):
    # combine a line with the next line if spoken by the same person
    # char = character to combine lines for (only done if fuzzy=True)
    max_idx = len(lines)
    new_lines = []
    cnt = 0

    while cnt < max_idx:
        if cnt == max_idx - 1:
            new_lines.append(lines[cnt])
            break

        if fuzzy:
            # fuzzy match required (e.g. "Ross"/"Ross" and also "Ross"/"Ross and Joey")
            speaker_is_char = check_if_speaker_is(lines[cnt], char) is True
            next_speaker_is_char = check_if_speaker_is(lines[cnt + 1], char) is True
            speakers_are_the_same = (
                True if speaker_is_char and next_speaker_is_char else False
            )
        else:
            # exact match required (e.g. "Ross"/"Ross" but not "Ross"/"Ross and Joey")
            speaker = speaker_is(lines[cnt])
            next_speaker = speaker_is(lines[cnt + 1])
            speakers_are_the_same = True if speaker == next_speaker else False

        if speakers_are_the_same:
            first_line = lines[cnt]
            next_line = get_spoken_lines(lines[cnt + 1])  # remove speaker name
            combined_lines = " ".join([first_line, next_line])
            new_lines.append(combined_lines)
            cnt += 2  # skip the next line
        else:
            new_lines.append(lines[cnt])
            cnt += 1
    if verbose:
        print(
            f"line compression for: {char} | original # lines: {max_idx} | new # of lines: {len(new_lines)}"
        )

    return new_lines


def add_payload(sample):
    sample["payload"] = config.TEMPLATE.format(
        prompt=sample["input"],
        response=sample["response"],
    )
    return sample


def tokenize_batch(batch, tokenizer, max_length):
    return tokenizer(batch["payload"], max_length=max_length, truncation=True)


def raw_dataset():
    # returns list of dicts, each dict containing input/response pairs, e.g.:
    # {
    #     "input_speaker": "Monica",
    #     "input": "There's nothing to tell! He's just some guy I work with!",
    #     "response_speaker": "Joey",
    #     "response": "C'mon, you're going out with the guy! There's gotta be something wrong with him!",
    # }
    filepath = f"{config.DIR}/{config.LINES_FILENAME}"
    print(f"reading data from: {filepath}")
    lines = open(f"{filepath}", "r").read().split("\n")

    # CLEANUP
    # notes:
    # 1) data prep does not separate a conversation if the speaker is the last one to speak in the episode and the first one to speak in the next episode
    # 2) recommend NOT performing "fuzzy" line compression given problems with lines spoken by "All" or equivalent (e.g. Joey/Chandler/Phoebe/Rachel/Monica/Ross)
    lines = [
        remove_writer_notes(remove_extra_spaces(cleanup_speakers((line.strip()))))
        for line in lines
        if len(line) > 0
        and ":" in line
        and line[0] not in ("[", "(", "{")
        and line[-1] not in ("]", ")", "}")
        and "written by:" not in line.lower()
        and "teleplay by" not in line.lower()
    ]
    for _ in range(4):
        lines = compress_lines(lines, None, fuzzy=False, verbose=True)
    pairs = [
        {
            "input_speaker": speaker_is(lines[i]),
            "input": get_spoken_lines(lines[i]),
            "response_speaker": speaker_is(lines[i + 1]),
            "response": get_spoken_lines(lines[i + 1]),
        }
        for i in range(len(lines) - 2)
    ]

    return pairs


def sft_dataset__joey(pairs, tokenizer, test_size=None, cp_list=None, return_df=False):
    """
    Unique dataset for joey.  Prioritizes lines that:
    # 1) include "joey" in the input line
    # 2) are from Chandler (his roommate who is likely asking conversing 1:1 with him)
    # 3) are about acting, auditioning (Joey's primary "job")
    # 4) have a question mark (decent indicator that a character will be directly responding)

    # After performing manual inspection, quality pairs (inputs/responses) increased
    from ~70% to ~75%
    """

    CHAR = "joey"

    char_data = [p for p in pairs if CHAR in p["response_speaker"].lower()]
    df = pd.DataFrame.from_records(char_data)

    # custom filters
    df.loc[:, "is_chandler_flag"] = [
        True if "chan" in speaker.lower() else False
        for speaker in df.input_speaker.values
    ]
    df.loc[:, "hasqm"] = [True if "?" in input else False for input in df.input.values]
    df.loc[:, "joey_in_line"] = [
        True if re.search(f"(?i){CHAR}", line) else False for line in df.input.values
    ]
    df.loc[:, "is_about_acting"] = [
        True if re.search("(?i)play|audition|acting|actor", line) else False
        for line in df.response.values
    ]
    df = df[
        (df.is_chandler_flag) | (df.joey_in_line) | (df.hasqm) | (df.is_about_acting)
    ]

    if cp_list is not None:
        df = df.loc[cp_list, :]

    if return_df:
        return df

    dataset = Dataset.from_pandas(df[["input", "response"]]).map(add_payload)

    # map() cannot accept a func requiring multiple params, so we create a partial to pre-fix the non-batch params
    tokenize_batch_partial_func = partial(
        tokenize_batch, tokenizer=tokenizer, max_length=config.MAX_LENGTH
    )

    dataset = dataset.map(
        tokenize_batch_partial_func,
        remove_columns=[
            "input",
            "response",
            "__index_level_0__",  # added by hf datasets
            # "payload",
        ],
    )

    dataset = dataset.filter(lambda x: len(x["input_ids"]) < config.MAX_LENGTH)
    if test_size is not None:
        dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=config.SEED)
    return dataset


def initialize_model_and_tokenizer():
    if not os.path.exists(config.SINGLE_MODEL_DIR):
        print(
            f"local model and tokenizer not found at {config.SINGLE_MODEL_DIR}, downloading from HF..."
        )
        Path(config.SINGLE_MODEL_DIR).mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=config.BNB_CONFIG,
            device_map="auto",
        )
        # model_mem = model.get_memory_footprint() / 1e9
        # print(f"MODEL SIZE: {model_mem:0.2f}GB")

        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        # add EOS token to end of sequence:
        tokenizer.add_eos_token = True if "llama" in config.MODEL_NAME.lower() else False

        print(f"saving model,tokenizer at: {config.SINGLE_MODEL_DIR}")
        model.save_pretrained(config.SINGLE_MODEL_DIR)
        tokenizer.save_pretrained(config.SINGLE_MODEL_DIR)
    else:
        print(f"loading model from local dir: {config.SINGLE_MODEL_DIR}")
        model = AutoModelForCausalLM.from_pretrained(config.SINGLE_MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(config.SINGLE_MODEL_DIR)

    return model, tokenizer


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable %: {100 * trainable_params / all_param:0.2f}%"
    )


def find_all_linear_names(model, check4bit, check8bit, verbose=False):
    """
    Find modules to apply LoRA to.
    `model` must be PEFT model
    """

    if check4bit == check8bit:
        raise ValueError("check4bit and check8bit cannot both have same value")
    
    if check4bit:
        cls = bnb.nn.Linear4bit
    elif check8bit:
        cls = bnb.nn.Linear8bitLt

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    if verbose:
        print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)
