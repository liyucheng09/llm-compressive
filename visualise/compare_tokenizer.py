import json
import numpy as np

with open('results/wikitext_results.json') as f:
    results = json.load(f)

models = ['Mistral-7B', 'Llama-2-7B-HF', 'Baichuan2-7B-Base', 'Qwen-7B', 'chatglm3-6b-base']
months_in_2023 = [f'2023-{month:02d}' for month in range(1, 12)]

metrics_for_tokenizer = {}

for model_name, data in results.items():
    if model_name not in models:
        continue

    tokens_months = []
    bpt_months = []
    bpb_months = []

    for month, metrics in data.items():
        if month not in months_in_2023:
            continue

        # this is in bytes
        compressed_size = metrics['compressed_size']

        # these two are in bits
        bpt = metrics['bpt']
        bpb = metrics['bpb']

        num_tokens = compressed_size * 8 / bpt

        tokens_months.append(num_tokens)
        bpt_months.append(bpt)
        bpb_months.append(bpb)

    total_tokens = sum(tokens_months)
    total_bpt = np.mean(bpt_months)
    total_bpb = np.mean(bpb_months)

    if model_name not in metrics_for_tokenizer:
        metrics_for_tokenizer[model_name] = {}

    metrics_for_tokenizer[model_name]['vocab_size'] = '-'
    metrics_for_tokenizer[model_name]['total_tokens'] = total_tokens
    metrics_for_tokenizer[model_name]['total_bpt'] = total_bpt
    metrics_for_tokenizer[model_name]['total_bpb'] = total_bpb

import pandas as pd

df = pd.DataFrame.from_dict(metrics_for_tokenizer, orient='index')
print(df.to_latex(float_format='%.4f'))