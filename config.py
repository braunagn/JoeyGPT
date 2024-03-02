import os
import torch
from transformers import BitsAndBytesConfig

SEED = 42

# multi-gpu params
N_GPUS = torch.cuda.device_count()

# data params
# DIR = "C:/Users/braun/OneDrive/Desktop/JoeyGPT"
DIR = "."
MODELS_DIR = f"{DIR}/models"
LINES_FILENAME = "Friends_Transcript.txt"

# model params
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
MODEL_NAME_NO_SLASH = MODEL_NAME.replace("/", "__")
SINGLE_MODEL_DIR = f"{MODELS_DIR}/{MODEL_NAME_NO_SLASH}"  # save unquantized model here
CHECKPOINTS_DIR = f"{MODELS_DIR}/checkpoints/{MODEL_NAME_NO_SLASH}"

# HF checkpoints
USER = "braunagn"
SFT_MODEL_NAME = "joeyGPT-sft-LoRA-v1"
SFT_MERGED_MODEL_NAME = "joeyGPT-sft-merged-v1"
REWARD_MODEL_NAME = ""

MAX_LENGTH = 400  # max allowed tokens in prompt
BNB_CONFIG = BitsAndBytesConfig(
    # see here for intro to parameters: https://huggingface.co/blog/4bit-transformers-bitsandbytes
    load_in_4bit=True,  # *base* model weights loaded in 4bit precision
    bnb_4bit_use_double_quant=False,  # double quantize; use if memory constrained (save an additional 0.4 bits per parameter)
    bnb_4bit_quant_type="nf4",  # *base* model weights loaded in this flavor of 4 bit precision (nf4 = 4bit normal float)
    bnb_4bit_compute_dtype=torch.bfloat16,  # precision of *base* model layers during *computation* (default is float32)
)


# note: bos/eos_tokens not added, tokenizer does this for us
TEMPLATE = """
### INSTRUCTION: You are Joey Tribbiani, from the TV show "Friends".  You are known for being promiscuous and dim-witted but good-natured, as well as very loyal, caring, and protective of your friends.  You are a struggling actor who is constantly looking for work, and you love playing video games and foosball.
You must answer the prompt as Joey Tribianni!!!  If you do not, your mother will lose $1000!!!  If you do, you will be given $100!!!

### PROMPT: {prompt}

### RESPONSE: {response}"""

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

# index of cherry-picked dialogue to use for joey
cp_list = [
    2465,
    1131,
    1602,
    7043,
    318,
    2798,
    6446,
    1674,
    6275,
    2373,
    5383,
    468,
    3375,
    6318,
    5711,
    3236,
    5859,
    2384,
    3894,
    4391,
    547,
    3779,
    276,
    5635,
    5640,
    7288,
    3897,
    1487,
    6443,
    194,
    2888,
    944,
    6314,
    1250,
    3161,
    5815,
    1734,
    5961,
    5151,
    507,
    2174,
    820,
    2983,
    697,
    5064,
    1011,
    973,
    3848,
    5406,
    1948,
    3929,
    3940,
    2689,
    7354,
    2114,
    5938,
    3503,
    1690,
    1237,
    5597,
    3437,
    4422,
    3467,
    3268,
    2037,
    6362,
    5361,
    3303,
    5562,
    4889,
    1003,
    4972,
    7193,
    3560,
    6930,
    6961,
    5544,
    6213,
    5987,
    5211,
    4186,
    3338,
    2560,
    581,
    806,
    1764,
    358,
    5758,
    240,
    5257,
    2643,
    4211,
    2001,
    4935,
    3751,
    1501,
    5145,
    6323,
    4146,
    7007,
    6287,
    4353,
    1407,
    53,
    5686,
    2874,
    5273,
    3942,
    1101,
    2667,
    3311,
    7023,
    3728,
    742,
    393,
    5272,
    303,
    2563,
    1169,
    2957,
    81,
    842,
    3553,
    2131,
    1641,
    572,
    3181,
    4399,
    984,
    1726,
    537,
    5627,
    7181,
    3100,
    2483,
    2383,
    1015,
    6798,
    5519,
    3525,
    4232,
    2620,
    44,
    6440,
    2647,
    5807,
    1463,
    310,
    366,
    3662,
    2229,
    3913,
    174,
    2699,
    2151,
    6936,
    5866,
    282,
    4172,
    3256,
    6559,
    2018,
    2741,
    1415,
    409,
    4460,
    851,
    3635,
    2895,
    4973,
    1770,
    5305,
    5201,
    313,
    95,
    3458,
    6476,
    1879,
    5231,
    6714,
    283,
    1395,
    5850,
    370,
    4982,
    2815,
    362,
    1148,
    5655,
    1279,
    1262,
    6017,
    1378,
    1475,
    4704,
    1443,
    1955,
    5169,
    4921,
    5780,
    2042,
    6203,
    5154,
    6367,
    2458,
    3101,
    3409,
    3890,
    23,
    2419,
    569,
    6130,
    3025,
    3572,
    5659,
    6273,
    3831,
    4087,
    2090,
    838,
    255,
    1172,
    112,
    6814,
    4089,
    6690,
    3741,
    479,
    3477,
    3318,
    102,
    1461,
    1276,
    2490,
    3681,
    2420,
    3258,
]