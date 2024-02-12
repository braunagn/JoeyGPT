import os
import torch
from transformers import BitsAndBytesConfig

SEED = 42

# multi-gpu params
N_GPUS = torch.cuda.device_count()

# data params
# DIR = "C:/Users/braun/OneDrive/Desktop/JoeyGPT"
DIR = "~/JoeyGPT"
MODELS_DIR = f"{DIR}/models"
LINES_FILENAME = "Friends_Transcript.txt"

# model params
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_NAME_NO_SLASH = MODEL_NAME.replace("/", "__")
SINGLE_MODEL_DIR = f"{MODELS_DIR}/{MODEL_NAME_NO_SLASH}"
MAX_LENGTH = 4000  # of prompt
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# note: bos/eos_tokens not added, tokenizer does this for us
TEMPLATE = """
### INSTRUCTION: You are Joey Tribbiani, from the TV show "Friends".  You are known for being promiscuous and dim-witted but good-natured, as well as very loyal, caring, and protective of your friends.  You are a struggling actor who is constantly looking for work, and you love playing video games and foosball.
You must answer the prompt as Joey Tribianni!!!  If you do not, your mother will lose $1000!!!  If you do, you will be given $100!!!

### PROMPT: {prompt}

### RESPONSE: {response}
"""

SPEAKER_REPLACEMENTS = [
    {
        "replace": [
            "Joey's voice, but she sees Ross:",
            "Joey's voice/Ross:",
            "Joey (to Rachel):",
            "Joey (to Chandler):",
            "Joey (to Ross):",
        ],
        "with": "Joey:",
    },
    {
        "replace": [
            "Joey's Look-A-Like:",
        ],
        "with": "J's Look-A-Like:",
    },
    {
        "replace": [
            "Joey's Date:",
            "Joeys Date:",
        ],
        "with": "J's Date:",
    },
    {
        "replace": [
            "Joeys Sisters:",
            "Joeys Sister:",
        ],
        "with": "J's Sister:",
    },
    {
        "replace": [
            "Joey's Co-Star:",
            "JOEY'S CO-STAR:",
        ],
        "with": "J's Co-Star:",
    },
    {
        "replace": [
            "Joey's Doctor:",
        ],
        "with": "J's Doctor:",
    },
    {
        "replace": [
            "Joey's Hand Twin:",
        ],
        "with": "J's Hand Twin:",
    },
    {
        "replace": [
            "Joeys Grandmother:",
        ],
        "with": "J's Grandmother:",
    },
    {
        "replace": [
            "Monica (to Joey):",
            "Monica (as Rachel):",
            "Monica (to Ross):",
            "Monica to Ross:",
            "Monica screaming at Ross:",
            "MNCA:",
        ],
        "with": "Monica:",
    },
    {
        "replace": [
            "PHOE/MNCA:",
            "PHOE/Monica:",
        ],
        "with": "Phoebe/Monica:",
    },
    {
        "replace": [
            "Phoebe (to Joey):",
            "Phoebe (to Rachel):",
            "Phoebe (walking to Ross carrying a black leather jacket):",
            "Phoebe-Estelle:",
        ],
        "with": "Phoebe:",
    },
    {
        "replace": [
            "Chandler (Stands up and walks to Joey):",
            "Chandler (to Joey):",
            "Chandler (to Rachel):",
            "Chandler (to Monica):",
            "Chandlers:",
            "Present Chandler's voice:",
        ],
        "with": "Chandler:",
    },
    {
        "replace": [
            "Janine [to Chandler]:",
        ],
        "with": "Janine:",
    },
    {
        "replace": [
            "Amy walks over to the couch and sits down next to Rachel:",
            "Amy turns to Ross and Rachel:",
            "Amy turns around to Phoebe:",
        ],
        "with": "Amy:",
    },
    {
        "replace": [
            "Rachel (as Monica):",
            "Rachel turns to Ross:",
            "Racel:",
            "Rache:",
            "RACH:",
        ],
        "with": "Rachel:",
    },
    {
        "replace": [
            "Rachels Boss:",
        ],
        "with": "Rs Boss:",
    },
    {
        "replace": [
            "Steve (staring at Rachel):",
        ],
        "with": "Steve:",
    },
    {
        "replace": [
            "Ross to Monica:",
            "Ross first has a look of 'huh' then changes it to sarcastic happy:",
        ],
        "with": "Ross:",
    },
    {
        "replace": [
            "Benjamin (to Ross):",
        ],
        "with": "Benjamin:",
    },
    {
        "replace": [
            "Phoebe Sr:",
            "Phoebe Sr.:",
        ],
        "with": "P-Sr:",
    },
    {
        "replace": [
            "Phoebe's Friends:",
        ],
        "with": "P-Friends:",
    },
    {
        "replace": [
            "Phoebe's Assistant:",
        ],
        "with": "P-Assistant:",
    },
    {
        "replace": [
            "Monica's Boyfriend:",
        ],
        "with": "M-Boyfriend:",
    },
    {
        "replace": [
            "All:",
            "ALL:",
        ],
        "with": "EVERYONE:",
    },
    {
        "replace": [
            "All (except Rachel):",
        ],
        "with": "Chandler/Joey/Ross/Monica/Phoebe:",
    },
    {
        "replace": [
            "Everyone almost simultaneously except Ross:",
        ],
        "with": "Chandler/Joey/Rachel/Monica/Phoebe:",
    },
    {
        "replace": [
            "Everyone but Monica:",
        ],
        "with": "Chandler/Joey/Ross/Rachel/Phoebe:",
    },
    {
        "replace": [
            "Mr. Treeger::",
        ],
        "with": "Mr. Treeger:",
    },

]

