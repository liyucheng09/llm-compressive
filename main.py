from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from data_processor import (
    BBCNewsProcessor,
    BBCImageProcessor,
    WikiTextProcessor,
    CodeProcessor,
    ArxivProcessor,
    AudioProcessor,
    MathProcessor,
)
import os
import sys
from tqdm import tqdm
from evaluator import Metrics
from auto_gptq import exllama_set_max_input_length
import time as time_module

model_max_context = {
    'Baichuan2-7B-Base': 4096,
    'chatglm3-6b-base': 32768,
    'internlm-7B': 2048,
    'LLaMA-13B': 2048,
    'Llama-2-13B': 4096,
    'Llama-2-70B': 4096,
    'Llama-2-7B': 4096,
    'Llama-2-7B-HF': 4096,
    'LLaMA-30B': 2048,
    'LLaMA-65B': 2048,
    'LLaMA-7B': 2048,
    'LLaMA-7B-HF': 2048,
    'Qwen-7B': 32768,
    'Yi-34B-200K': 200000,
    'Yi-6B': 4096,
    'Yi-6B-200K': 200000,
    'Mistral-7B': 32768,
    'Mistral-7B-Instruct': 32768,
    'Qwen/Qwen1.5-7B': 32768,
    'google/gemma-7b': 8192,
}

def prepare_data(data_name, save_path, tokenizer):
    all_time_stamps = [f'{year}-{month:02d}' for year in range(2017, 2024) for month in range(1, 13) if not (year == 2023 and month > 11)][53:]

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
    elif data_name == 'code':
        data_path = 'RealTimeData/code_alltime'
        modality = 'text'
        processor = CodeProcessor
    elif data_name == 'arxiv':
        data_path = 'RealTimeData/arxiv_alltime'
        modality = 'text'
        processor = ArxivProcessor
    elif data_name == 'audio':
        data_path = 'RealTimeData/audio_alltime'
        modality = 'audio'
        processor = AudioProcessor
    elif data_name == 'math':
        data_path = 'RealTimeData/math_alltime'
        modality = 'text'
        processor = MathProcessor

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
    elif 'gptq' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True, attn_implementation="flash_attention_2")
    elif 'yi' in model_name.lower() or 'mistral' in model_name.lower() or 'llama' in model_name.lower() or 'gemma' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True, attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)

    return model, tokenizer

if __name__ == '__main__':
    model_name, data_name, save_path, context_size, batch_size, = sys.argv[1:]
    if 'qwen' in model_name.lower():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    batch_size = int(batch_size)
    if context_size == 'stride':
        context_size = 2048
        stride = 512
    elif context_size == 'max_length':
        # restrict the context size up to 8192 to prevent OOM
        context_size = min(model_max_context[model_name], 12288)
        stride = None
    else:
        context_size = int(context_size)
        stride = None
    
    # where is your model? use your path here.
    # Set model_path = model_name if you want to download the model from huggingface
    model_path = os.path.join('/mnt/fast/nobackup/scratch4weeks/yl02706/models', model_name)
    if not os.path.exists(model_path):
        model_path = model_name
    model, tokenizer = load_model_and_tokenizer(model_path)

    # resize buffer size for exllama, if gptq is used
    if getattr(model.config, 'quantization_config', None) is not None and model.config.quantization_config.use_exllama and model.config.quantization_config.desc_act:
        model = exllama_set_max_input_length(model, context_size*batch_size)
        
    all_data, modality = prepare_data(data_name, save_path, tokenizer)
    print(f'Data {data_name}, Modality {modality}')
    time_used = 0

    for data in all_data:
        name, time = data.name, data.config
        print(f'Processing {name} {time}...')

        data.prepare_batches(context_size, stride = stride)
        print(f'Total number of chunks: {data.metadata["num_chunks"]}')

        metrics = Metrics(modality, save_path, model_name, byte2id=data.byte2ids if modality != 'text' else None, use_arithmetic_coding=False)
        
        start_time = time_module.time()
        for i, chunk in enumerate(tqdm(data.batches(batch_size))):

            input_ids = torch.tensor(chunk, dtype=torch.long, device=model.device)
            input_ = {'input_ids': input_ids}
            with torch.no_grad():
                output = model(**input_)
            
            logits = output.logits

            metrics.step(logits, input_ids, stride = stride)
        
        time_used += time_module.time() - start_time
        data.metadata['time_used'] = time_used

        metrics(data.stream, data.metadata, model_name)
        print(f'==== Finished processing {name} {time}. Self-info: {metrics.self_info_cache} ======')

        metrics.clear_cache()