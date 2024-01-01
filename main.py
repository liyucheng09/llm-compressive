from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from data_processor import BBCProcessor
import os
import sys

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    if 'chatglm' in model_name.lower():
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    elif 'llama' in model_name.lower() or 'gptq' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)

    return model, tokenizer

if __name__ == '__main__':
    