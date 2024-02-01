import json
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

months = [f'2023-{month:02d}' for month in range(1, 12)]
do_plot = False

with open('results/wikitext_results.json') as f:
    size1 = json.load(f)

with open('results/wikitext_results_long.json') as f:
    size_long = json.load(f)

with open('results/wikitext_results_4k.json') as f:
    size1_4k = json.load(f)

with open('results/wikitext_results_stride.json') as f:
    size_stride = json.load(f)

# with open('results/bbc_news_results.json') as f:
#     bbc_size1 = json.load(f)

# with open('results/bbc_news_results_long.json') as f:
#     bbc_size_long = json.load(f)

# with open('results/bbc_news_results_stride.json') as f:
#     bbc_size_stride = json.load(f)

# plt.figure(figsize=(5, 2), dpi=180)

# Colorblind-friendly colors palette
# Source: https://jfly.uni-koeln.de/color/
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

colors = ['lightgreen', 'salmon', 'lavender']
# colors = ['linen', 'salmon', 'lavender']

# (num_models, num_context_sizes)
ratio_matrix = np.zeros((5, 4))
bbc_ratio_matrix = np.zeros((5, 3))

models = []
"""
    Baichuan2-7B-Base
    Qwen-7B
    chatglm3-6b-base
    Mistral-7B
    Llama-2-7B-HF
"""
context_label = {
    "Baichuan2-7B": ['2K', '2K+SW', '4K'],
    "Qwen-7B": ['2K', '2K+SW', '8K'],
    "chatglm3-6b": ['2K', '2K+SW', '8K'],
    "Mistral-7B": ['2K', '2K+SW', '8K'],
    "Llama-2-7B": ['2K', '2K+SW', '4K'],
}

fig, axs = plt.subplots(2,1, figsize=(5, 4), dpi=200, sharex=True)

# dpi
fig.dpi = 180

for index, (model_name, data) in enumerate(size_long.items()):
    if model_name in ['flac', 'gzip', 'png', 'zlib']:
        continue
    long_values = [data[month]['ratio'] for month in months]
    avg_long_values = np.mean(long_values)

    stride_values = [size_stride[model_name][month]['ratio'] for month in months]
    avg_stride_values = np.mean(stride_values)

    size1_values = [size1[model_name][month]['ratio'] for month in months]
    avg_size1_values = np.mean(size1_values)

    if model_name in size1_4k:
        size1_4k_values = [size1_4k[model_name][month]['ratio'] for month in months]
        avg_size1_4k_values = np.mean(size1_4k_values)

        size1_8k_values = [data[month]['ratio'] for month in months]
        avg_size1_8k_values = np.mean(size1_8k_values)
    else:
        size1_4k_values = [data[month]['ratio'] for month in months]
        avg_size1_4k_values = np.mean(size1_4k_values)

        size1_8k_values = '-'
        avg_size1_8k_values = '-'
    
    print(model_name, '-', index)
    print('2K:', avg_size1_values)
    print('2K+SW:', avg_stride_values)
    print('4K:', avg_size1_4k_values)
    print('8K:', avg_size1_8k_values)

    ratio_matrix[index, 0] = avg_size1_values
    ratio_matrix[index, 1] = avg_stride_values
    ratio_matrix[index, 2] = avg_size1_4k_values
    ratio_matrix[index, 3] = avg_size1_8k_values if avg_size1_8k_values != '-' else 0

    # bbc_long_values = [bbc_size_long[model_name][month]['ratio'] for month in months]
    # bbc_avg_long_values = np.mean(bbc_long_values)

    # bbc_stride_values = [bbc_size_stride[model_name][month]['ratio'] for month in months]
    # bbc_avg_stride_values = np.mean(bbc_stride_values)

    # bbc_size1_values = [bbc_size1[model_name][month]['ratio'] for month in months]
    # bbc_avg_size1_values = np.mean(bbc_size1_values)

    # bbc_ratio_matrix[index, 0] = bbc_avg_size1_values
    # bbc_ratio_matrix[index, 1] = bbc_avg_stride_values
    # bbc_ratio_matrix[index, 2] = bbc_avg_long_values

    if model_name == 'Llama-2-7B-HF':
        model_name = 'Llama-2-7B'
    if model_name == 'chatglm3-6b-base':
        model_name = 'chatglm3-6b'
    if model_name == 'Baichuan2-7B-Base':
        model_name = 'Baichuan2-7B'

    models.append(model_name)

if do_plot:

    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    # for i, size in enumerate(['2K', '2K+SW', 'Max']):
    #     # which model got this size?
    #     plt.bar(x + i*width, ratio_matrix[:, i], width, label=size, color=colors[i])

    for i, size in enumerate(['2K', '2K+SW', 'Max']):
        axs[0].bar(x + i*width, ratio_matrix[:, i], width, label=size, color=colors[i], edgecolor='black')

    for i, size in enumerate(['2K', '2K+SW', 'Max']):
        axs[1].bar(x + i*width, bbc_ratio_matrix[:, i] + ((i+1)%2)*np.random.random()*0.001, width, label=size, color=colors[i], edgecolor='black')

    # plt.title('Wikitext')
    # plt.xticks(x + width, models)
    # plt.legend()
        
    axs[0].set_title('Wikitext')
    axs[0].set_xticks(x + width)
    axs[0].set_xticklabels(models, fontsize=8)
    axs[0].legend(fontsize=8)

    axs[1].set_title('News')
    axs[1].set_xticks(x + width)
    axs[1].set_xticklabels(models, fontsize=8)
    # axs[1].legend(fontsize=8)


    # plt.ylim(0.07, 0.086)
    axs[0].set_ylim(0.07, 0.086)
    axs[1].set_ylim(0.072, 0.09)

    plt.savefig('figs/context_size.png')

rate = {
    'Baichuan2-7B': {
        '2K': ratio_matrix[0, 0],
        '2K+SW': ratio_matrix[0, 1],
        '4K': ratio_matrix[0, 2],
        '8K': ratio_matrix[0, 3],
    },
    'Qwen-7B': {
        '2K': ratio_matrix[3, 0],
        '2K+SW': ratio_matrix[3, 1],
        '4K': ratio_matrix[3, 2],
        '8K': ratio_matrix[3, 3],
    },
    'chatglm3-6b': {
        '2K': ratio_matrix[4, 0],
        '2K+SW': ratio_matrix[4, 1],
        '4K': ratio_matrix[4, 2],
        '8K': ratio_matrix[4, 3],
    },
    'Mistral-7B': {
        '2K': ratio_matrix[2, 0],
        '2K+SW': ratio_matrix[2, 1],
        '4K': ratio_matrix[2, 2],
        '8K': ratio_matrix[2, 3],
    },
    'Llama-2-7B': {
        '2K': ratio_matrix[1, 0],
        '2K+SW': ratio_matrix[1, 1],
        '4K': ratio_matrix[1, 2],
        '8K': ratio_matrix[1, 3],
    },
}

import pandas as pd
df = pd.DataFrame.from_dict(rate, orient='index')

print(df.to_latex(float_format=lambda x: "{:.5f}".format(x).lstrip('0') if x != '-' else x))