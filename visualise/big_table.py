import json
import numpy as np

with open('results/wikitext_results_v2.json') as f:
    wiki_results = json.load(f)

with open('results/bbc_news_results_v2.json') as f:
    bbc_results = json.load(f)

with open('results/bbc_image_results_v2.json') as f:
    bbc_image_results = json.load(f)

with open('results/code_results_v2.json') as f:
    code_results = json.load(f)

with open('results/arxiv_results_v2.json') as f:
    arxiv_results = json.load(f)

with open('results/audio_results.json') as f:
    audio_results = json.load(f)

model_name_to_label = {
    'Baichuan2-7B-Base': 'Baichuan2-7B',
    'internlm-7B': 'Internlm-7B',
    'Qwen-7B': 'Qwen-7B',
    'Yi-6B': 'Yi-6B',
    'chatglm3-6b-base': 'Chatglm3-6B',
    'Mistral-7B': 'Mistral-7B',
    'LLaMA-7B-HF': 'LLaMA-7B',
    'LLaMA-13B': 'LLaMA-13B',
    'Llama-2-13B': 'Llama-2-13B',
    'Llama-2-7B-HF': 'Llama-2-7B',
    'CodeLlama-7B': 'CodeLlama-7B',
    'Llama-2-70B': 'Llama-2-70B',
    'LLaMA-30B': 'LLaMA-30B',
    'LLaMA-65B': 'LLaMA-65B',
    'Yi-34B-200K': 'Yi-34B',
    'flac': 'Flac',
    'png': 'Png',
    'zlib': 'Zlib',
}

# 2017-2023, except 2023-12
all_months = [f'{year}-{month:02d}' for year in range(2017, 2024) for month in range(1, 13) if not (year == 2023 and month == 12)]

# 2023
months_in_2023 = [f'2023-{month:02d}' for month in range(1, 12)]

# pre-2023
pre_2023_months = [f'{year}-{month:02d}' for year in range(2017, 2023) for month in range(1, 13)]

models = {}

def compute_results(data, model, task):
    ratios_all_months = []
    for month, metrics in data.items():
        if month in all_months:
            ratios_all_months.append(metrics['ratio'])

    if len(ratios_all_months) < 75: print(model, task, len(ratios_all_months))
    avg_all_months = np.mean(ratios_all_months)

    ratios_2023 = [ data[month]['ratio'] for month in months_in_2023 if month in data]
    avg_2023 = np.mean(ratios_2023)

    pre_2023_avg = np.mean([ data[month]['ratio'] for month in pre_2023_months if month in data])
    diff_train_test = avg_2023 - pre_2023_avg

    diff = avg_2023 - avg_all_months

    def float_formatter(x, digits=4):
        if x <1:
            num_str = ('{:.' + str(digits) + 'f}').format(x).lstrip('0')
        else:
            num_str = ('{:.' + str(digits) + 'g}').format(x)
        
        # pad 0 if not enough digits
        num_digits = len(num_str.replace('.', ''))
        if num_digits < digits:
            digits_to_pad = digits - num_digits
            if '.' not in num_str:
                num_str += '.'
            num_str += '0' * digits_to_pad
        return num_str

    avg_all_months_str = '{:.4f}'.format(avg_all_months).lstrip('0')
    avg_all_months_str_100 = float_formatter(avg_all_months * 100)
    avg_2023_str = '{:.4f}'.format(avg_2023).lstrip('0')
    avg_2023_str_100 = float_formatter(avg_2023 * 100)
    pre_2023_avg_str = '{:.4f}'.format(pre_2023_avg).lstrip('0')

    arrow = '↑' if diff > 0 else '↓'
    diff_str = '{:.4f}'.format(abs(diff)).lstrip('0')
    avg_2023_with_diff = f'{avg_2023_str} {arrow} {diff_str}'

    arrow2 = '↑' if diff_train_test > 0 else '↓'
    diff_train_test_str = '{:.4f}'.format(abs(diff_train_test)).lstrip('0')
    diff_train_test_str_100 = float_formatter(abs(diff_train_test) * 100, digits=3)

    pre_2023_avg_with_diff = f'{avg_2023_str} {arrow2} {diff_train_test_str}'
    pre_2023_avg_with_diff_ = f'{avg_2023_str_100} {arrow2} {diff_train_test_str_100}'

    return {
        # 'Avg.': avg_all_months_str,
        'Avg.': avg_all_months_str_100,
        # '2023': pre_2023_avg_with_diff,
        '2023': pre_2023_avg_with_diff_,
        # # 'performance': avg_all_months,
        # 'train': pre_2023_avg_str,
        # 'test': pre_2023_avg_with_diff,
        # 'Avg.': avg_all_months,
        'performance': avg_2023 * 100,
        'robustness': diff_train_test * 100,
    }

for model_name, data in wiki_results.items():
    if model_name in ['LLaMA-7B', 'Llama-2-7B']:
        continue
    print(model_name)
    model_name_ = model_name_to_label[model_name]

    if model_name_ not in models:
        models[model_name_] = {}

    if model_name_ not in bbc_results:
        bbc_results[model_name_] = {}
    
    if model_name_ not in bbc_image_results:
        bbc_image_results[model_name_] = {}

    if model_name_ not in code_results:
        code_results[model_name_] = {}

    if model_name_ not in arxiv_results:
        arxiv_results[model_name_] = {}
    
    if model_name not in audio_results:
        audio_results[model_name] = {}

    wiki_results_ = compute_results(data, model_name, 'Wikitext')
    models[model_name_]['Wikitext'] = wiki_results_

    news_results = compute_results(bbc_results[model_name], model_name, 'BBC News')
    # models[model_name]['BBC News'] = news_results
    models[model_name_]['BBC News'] = news_results

    image_results = compute_results(bbc_image_results[model_name], model_name, 'BBC Image')
    # models[model_name]['Image'] = image_results
    models[model_name_]['Image'] = image_results

    code_results_ = compute_results(code_results[model_name], model_name, 'Code')
    # models[model_name]['Code'] = code_results_
    models[model_name_]['Code'] = code_results_

    arxiv_results_ = compute_results(arxiv_results[model_name], model_name, 'Arxiv')
    # models[model_name]['Arxiv'] = arxiv_results_
    models[model_name_]['Arxiv'] = arxiv_results_

    audio_results_ = compute_results(audio_results[model_name], model_name, 'Audio')
    models[model_name_]['Audio'] = audio_results_

do_latex = True
do_plot = False
task = 'Code'
fig_type = '7B'

if do_latex:
    import pandas as pd

    df = pd.DataFrame.from_dict({(i, j): models[i][j]
                                for i in models.keys()
                                for j in models[i].keys()},
                                orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack().swaplevel(0, 1, axis=1).sort_index(axis=1)

    # new_order = ['Avg.', '2023']
    new_order = ['Avg.', '2023']
    df = df.reindex(columns=[(col, subcol) for col in df.columns.levels[0] for subcol in new_order])

    top_level_order = ['Wikitext', 'BBC News', 'Code', 'Arxiv', 'Image', 'Audio']
    df = df[top_level_order]

    print(df)
    print(df.to_latex())

if do_plot:
    import matplotlib.pyplot as plt

    # Clamping robustness to a minimum of 0
    new_models = {}
    plt.figure(figsize=(8, 5), dpi=200)

    print('='*30)
    performances = []
    robustnesses = []
    for model in models:
        print(model)
        if model in ['flac', 'png', 'zlib']:
            continue
        # if not ('7b' not in model.lower() and '6b' not in model.lower()):
        if fig_type == '7B' and '7b' not in model.lower() and '6b' not in model.lower():
            continue
        if fig_type == 'large' and '13b' not in model.lower() and '34b' not in model.lower() and '70b' not in model.lower() and '65b' not in model.lower():
            continue
        # if 'llama' not in model.lower():
        #     continue
        models[model][task]['robustness'] = max(models[model][task]['robustness'], 0)
        new_models[model] = models[model]
        performances.append(models[model][task]['performance'])
        robustnesses.append(models[model][task]['robustness'])
    models = new_models

    max_performance = max(performances)
    min_performance = min(performances)
    max_robustness = max(robustnesses)
    min_robustness = min(robustnesses)

    performance_range = max_performance - min_performance
    robustness_range = max_robustness - min_robustness

    performance_padding = performance_range * 0.05  # Adding 5% padding
    robustness_padding = robustness_range * 0.05     # Adding 5% padding

    # Adjusting the axis limits with added padding
    plt.xlim(max_performance + performance_padding, min_performance - performance_padding)  # Inverting x-axis
    plt.ylim(max_robustness + robustness_padding, min_robustness - robustness_padding)    # Inverting y-axis

    mid_performance = (max_performance + min_performance) / 2
    mid_robustness = (max_robustness + min_robustness) / 2

    # Drawing lines to divide the plot into quadrants
    plt.axvline(x=mid_performance, color='grey', linestyle='--')
    plt.axhline(y=mid_robustness, color='grey', linestyle='--')

    # Plotting
    for model in models:
        plt.scatter(models[model][task]['performance'], models[model][task]['robustness'], label=model, marker='o')
        text_x_offset = 0.2
        text_y_offset = -0.002
        # if model == 'Baichuan2-7B':
        #     text_x_offset = 0.4
        #     text_y_offset = .004
        # if model == 'Yi-6B':
        #     text_x_offset = 0
        # Check if label is close to the right edge
        if models[model][task]['performance'] > max_performance - performance_padding:
            text_x_offset = 0  # Move text to the left

        plt.text(models[model][task]['performance'] + text_x_offset, models[model][task]['robustness'] + text_y_offset, model, fontsize=12)

    plt.xlabel(f'Compression Rate - {task} (%,  lower is better, axis inverted)', fontsize=12)
    plt.ylabel('Robustness (gap of train/test period, %)', fontsize=12)
    # plt.title('Model Performance vs Robustness')
    # plt.gca().invert_xaxis()  # Inverting x-axis to have the best models on top-right
    # plt.gca().invert_yaxis()  # Inverting y-axis as well
    plt.legend()
    # plt.legend(loc='lower left')
    # plt.legend(loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/robustness_performance_{task}_{fig_type}.png')