from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from data_processor import (
    BBCNewsProcessor,
    BBCImageProcessor,
    WikiTextProcessor,
)
import os
import sys
from tqdm import tqdm
from evaluator import Metrics
from auto_gptq import exllama_set_max_input_length

model_max_context = {
    'Baichuan2-7B-Base': 4096,
    'chatglm3-6b-base': 32768,
    'internlm-7B': 2048,
    'LLaMA-13B': 2048,
    'Llama-2-13B': 4096,
    'Llama-2-70B': 4096,
    'Llama-2-7B': 4096,
    'LLaMA-30B': 2048,
    'LLaMA-65B': 2048,
    'LLaMA-7B': 2048,
    'Qwen-7B': 32768,
    'Yi-34B-200K': 2048,
    'Yi-6B': 4096,
    'Mistral-7B': 32768,
    'Mistral-7B-Instruct': 32768,
}

def prepare_data(data_name, save_path, tokenizer):
    all_time_stamps = [f'{year}-{month:02d}' for year in range(2017, 2024) for month in range(1, 13) if not (year == 2023 and month > 11)]
    if data_name == 'bbc_news':
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
            tokenizer = tokenizer,
            config = time
        )
        for time in all_time_stamps
    ]
    return all_data, modality

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    if 'chatglm' in model_name.lower():
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    elif 'hf' not in model_name.lower() and ('llama' in model_name.lower() or 'Yi-34B' in model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True, attn_implementation="flash_attention_2")
    elif 'yi' in model_name.lower() or 'mistral' in model_name.lower() or 'hf' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True, attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)

    return model, tokenizer

if __name__ == '__main__':
    model_name, data_name, save_path, context_size, batch_size, = sys.argv[1:]
    batch_size = int(batch_size)
    if context_size == 'max_length':
        # restrict the context size up to 8192 to prevent OOM
        context_size = min(model_max_context[model_name], 8192)
    else:
        context_size = int(context_size)
    
    model_path = os.path.join('/mnt/fast/nobackup/scratch4weeks/yl02706/models', model_name)
    model, tokenizer = load_model_and_tokenizer(model_path)
    if getattr(model.config, 'quantization_config', None) is not None and model.config.quantization_config.use_exllama and model.config.quantization_config.desc_act:
        model = exllama_set_max_input_length(model, context_size*batch_size)
    all_data, modality = prepare_data(data_name, save_path, tokenizer)
    print(f'Data {data_name}, Modality {modality}')

    if 'qwen' in model_name.lower() and modality != 'text':
        raise ValueError('Qwen do not support byte tokenization. So text data only.')

    for data in all_data:
        name, time = data.name, data.config
        print(f'Processing {name} {time}...')

        data.prepare_batches(context_size)
        print(f'Total number of chunks: {data.metadata["num_chunks"]}')

        metrics = Metrics(modality, save_path, model_name, byte2id=data.byte2ids if modality != 'text' else None)
        
        for i, chunk in enumerate(tqdm(data.batches(batch_size))):

            input_ids = torch.tensor(chunk, dtype=torch.long, device=model.device)
            input_ = {'input_ids': input_ids}
            with torch.no_grad():
                output = model(**input_)
            
            logits = output.logits

            metrics.step(logits, input_ids)

        metrics(data.stream, data.metadata, model_name)
        print(f'==== Finished processing {name} {time}.======')

        metrics.clear_cache()

