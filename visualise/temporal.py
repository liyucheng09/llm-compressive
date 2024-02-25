import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.ticker as ticker

task = 'math'
with open(f'results/{task}_results.json') as f:
    data = json.load(f)

small = True
fig_type = '7B'
code_llama = False
large_llama = False
x_ticks = True
no_ylabel = False
internlm = False
small_llama=False
glm = False
mistral = True
long_context = True
smoothing_ratio = None

# Create a plot
if not small:
    plt.figure(figsize=(20, 10), dpi=160)
    markersize = 6
    legend_fontsize = 17
    tick_fontsize = 15
    label_fontsize = 16
else:
    if no_ylabel:
        plt.figure(figsize=(8, 5), dpi=180)
    else:
        plt.figure(figsize=(8.2, 5), dpi=180)
    markersize = 3
    legend_fontsize = 10
    tick_fontsize = 12
    label_fontsize = 14

def exponential_smoothing(data, alpha = 0.5):
    """
    Apply exponential smoothing to the data.
    :param data: List of data points.
    :param alpha: Smoothing factor, between 0 and 1.
    :return: List of smoothed data points.
    """
    smoothed_data = []
    for i, point in enumerate(data):
        if i == 0:
            # The first smoothed value is the first data point.
            smoothed_data.append(point)
        else:
            # Compute the smoothed value.
            new_smoothed = alpha * point + (1 - alpha) * smoothed_data[i-1]
            smoothed_data.append(new_smoothed)
    return smoothed_data

# Colorblind-friendly colors palette
# Source: https://jfly.uni-koeln.de/color/
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

# Different line styles
line_styles = ['-', '--', '-.', ':']

# Different markers
markers = ['o', 's', 'D', '^', 'v', '<', '>']

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
    'Qwen1.5-7B': 'Qwen1.5-7B',
    'Qwen-7B-12288-long': 'Qwen-7B-12K',
    'Qwen1.5-7B-12288-long': 'Qwen1.5-7B-12K',
}

# Loop through each model's data and plot it
counter = 0
for model_name, model_data in data.items():
    if model_name not in model_name_to_label:
        continue
    if not long_context and 'long' in model_name:
        continue
    if fig_type == 'llama':
        if not code_llama and model_name == 'CodeLlama-7B':
            continue
        if not large_llama and model_name in ['Llama-2-70B', 'LLaMA-30B', 'LLaMA-65B']:
            continue
        if model_name not in ['Llama-2-7B-HF', 'LLaMA-7B-HF', 'LLaMA-13B', 'Llama-2-13B', 'CodeLlama-7B', 'Llama-2-70B', 'LLaMA-30B', 'LLaMA-65B']:
            continue
    elif fig_type == '7B':
        if not ('7b' in model_name.lower() or '6b' in model_name.lower()):
            continue
        if not internlm and 'internlm' in model_name.lower():
            continue
        if not code_llama and 'code' in model_name.lower():
            continue
        if not small_llama and ('LLaMA-7B-HF' in model_name or 'Llama-2-7B' in model_name):
            continue
        if not glm and 'chatglm3' in model_name.lower():
            continue
        if not mistral and 'mistral' in model_name.lower():
            continue

    else:
        # model_specific ploting
        if fig_type not in model_name.lower():
            continue
    
    labels = list(model_data.keys())
    values = np.array([metrics['ratio'] for metrics in model_data.values()]) * 100

    # remove 2021-03
    remove_index = labels.index('2021-03')
    values[remove_index] = (values[remove_index - 1] + values[remove_index + 1]) / 2
    
    if smoothing_ratio is not None:
        values = exponential_smoothing(values, alpha=smoothing_ratio)
    
    # Use color and line style from our predefined lists
    color = colors[counter % len(colors)]
    line_style = line_styles[counter % len(line_styles)]
    marker = markers[counter % len(markers)]
    
    plt.plot(labels, values, color=color, linestyle=line_style, marker=marker, label=model_name_to_label[model_name], markersize=markersize)
    counter += 1

# Adding title and labels
if small:
    plt.title(f'{task}')
    if not no_ylabel:
        plt.ylabel(f'Compression Rates (%)', fontsize=label_fontsize)
else:
    if not no_ylabel:
        plt.ylabel(f'Compression Rates - {task}', fontsize=label_fontsize)

# Adding a legend to differentiate the lines from each model
# plt.legend(fontsize=legend_fontsize)
plt.legend(fontsize=legend_fontsize, loc=(0.01, 0.01))
# plt.legend(fontsize=legend_fontsize, loc=(0.82,0.73))

# Display the plot
plt.grid(True)
# plt.xticks(rotation=45, ha='right')

# x ticks are too dense, so we only show every 3rd tick
# plt.xticks(list(range(0, len(labels), 6)), rotation=45, ha='right')
if small:
    plt.xticks(list(range(0, len(labels), 6)), rotation=45)
else:
    plt.xticks(list(range(0, len(labels), 2)), rotation=45, ha='right')

if not x_ticks:
    plt.xticks([])

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.4))
plt.tight_layout()

plt.tick_params(axis='x', labelsize=tick_fontsize)
plt.tick_params(axis='y', labelsize=tick_fontsize)

plt.savefig(f'figs/{fig_type}-{task}.png')