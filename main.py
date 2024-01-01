from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from data_processor import (
    BBCNewsProcessor,
    BBCImageProcessor,
    WikiTextProcessor,
)
import os
import sys

def prepare_data(data_name, save_path):
    all_time_stamps = [f'{year}-{month:02d}' for year in range(2017, 2024) for month in range(1, 13)]
    if data_name == 'bbc':
        data_path = 'RealTimeData/bbc_news_alltime'
        modality = 'text'
        processor = BBCNewsProcessor
    elif data_name == 'wikitext':
        data_path = 'RealTimeData/wikitext_alltime'
        modality = 'text'
        processor = WikiTextProcessor
    elif data_name == 'bbc_image':
        data_path = 'RealTimeData/bbc_images_alltime'
        modality = 'image'
        processor = BBCImageProcessor
    
    all_data = [
        processor(
            name = data_name,
            modality = modality,
            load_path = data_path,
            cache_path = os.path.join(save_path, data_name),
            config = time
        )
        for time in all_time_stamps
    ]
    return all_data

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
    model_name, data_name, save_path, = sys.argv[1:]

    all_data = prepare_data(data_name, save_path)
    model, tokenizer = load_model_and_tokenizer(model_name)