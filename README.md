# JoeyGPT
*Turn a pre-trained GPT into a chatbot that sounds like Joey from the TV show Friends.*

This project shows how to use commodity GPUs (e.g. RTX A4000s), quantization (e.g. 4bit models) and PEFT traning techniques (LoRA) to turn a base GPT model into a chatbot that sounds like Joey from the TV show Friends.

## Details

- Base GPT Model: [Mistral 7B V1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- quantization method: [4bit normal float](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- PEFT technique: [LoRA](https://arxiv.org/abs/2106.09685)
- Packages: see requirements.txt
