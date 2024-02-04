import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.ticker as ticker

task = 'wikitext'
with open(f'results/{task}_results.json') as f:
    data = json.load(f)

# we have two fig_types now: 'llama' and '7B'
fig_type = '7B'

# fine-grained control over which models to include
code_llama = True
large_llama = False
internlm = False

x_ticks = True
no_ylabel = False

# Create the plot, and set the font sizes
if no_ylabel:
    plt.figure(figsize=(8, 5), dpi=180)
else:
    plt.figure(figsize=(8.2, 5), dpi=180)
markersize = 3
legend_fontsize = 10
tick_fontsize = 12
label_fontsize = 14

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
}

# Loop through each model's data and plot it
counter = 0
for model_name, model_data in data.items():
    if model_name not in model_name_to_label:
        continue
    if fig_type == 'llama':
        if not code_llama and model_name == 'CodeLlama-7B':
            continue
        if not large_llama and model_name in ['Llama-2-70B', 'LLaMA-30B', 'LLaMA-65B']:
            continue
        if model_name not in ['Llama-2-7B-HF', 'LLaMA-7B-HF', 'LLaMA-13B', 'Llama-2-13B', 'CodeLlama-7B', 'Llama-2-70B', 'LLaMA-30B', 'LLaMA-65B']:
            continue
    elif fig_type == '7B':
        if not ('7B' in model_name or '6B' in model_name):
            continue
        if task not in ['code', 'bbc_image', 'arxiv']:
            if 'internlm' in model_name.lower() and not internlm:
                continue
            if 'code' in model_name.lower() and not code_llama:
                continue
    else:
        raise ValueError(f'Unknown fig_type: {fig_type}')
    
    labels = list(model_data.keys())
    values = np.array([metrics['ratio'] for metrics in model_data.values()]) * 100

    # remove 2021-03
    remove_index = labels.index('2021-03')
    values[remove_index] = (values[remove_index - 1] + values[remove_index + 1]) / 2
    
    # Use color and line style from our predefined lists
    color = colors[counter % len(colors)]
    line_style = line_styles[counter % len(line_styles)]
    marker = markers[counter % len(markers)]
    
    plt.plot(labels, values, color=color, linestyle=line_style, marker=marker, label=model_name_to_label[model_name], markersize=markersize)
    counter += 1

# Adding title and labels
plt.title(f'{task}')
if not no_ylabel:
    plt.ylabel(f'Compression Rates (%)', fontsize=label_fontsize)

# Adding a legend to differentiate the lines from each model
plt.legend(fontsize=legend_fontsize)

# Display the plot
plt.grid(True)

# x ticks are too dense, so we only show every 3rd tick
plt.xticks(list(range(0, len(labels), 6)), rotation=45)

if not x_ticks:
    plt.xticks([])

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.4))
plt.tight_layout()

plt.tick_params(axis='x', labelsize=tick_fontsize)
plt.tick_params(axis='y', labelsize=tick_fontsize)

plt.savefig(f'figs/{fig_type}-{task}.png')