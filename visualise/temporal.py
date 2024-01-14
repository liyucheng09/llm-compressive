import matplotlib.pyplot as plt
import json
import numpy as np

with open('results/code_results.json') as f:
    data = json.load(f)

# Create a plot
plt.figure(figsize=(25, 12), dpi=150)

# Colorblind-friendly colors palette
# Source: https://jfly.uni-koeln.de/color/
colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

# Different line styles
line_styles = ['-', '--', '-.', ':']

# Different markers
markers = ['o', 's', 'D', '^', 'v', '<', '>']

# Loop through each model's data and plot it
counter = 0
for model_name, model_data in data.items():
    # if 'llama' not in model_name.lower():
    #     continue
    # if '70' in model_name.lower():
    #     continue
    if '7B' not in model_name:
        continue
    if 'intern' in model_name.lower():
        continue
    
    labels = list(model_data.keys())
    values = np.array([metrics['ratio'] for metrics in model_data.values()]) * 100
    
    # Use color and line style from our predefined lists
    color = colors[counter % len(colors)]
    line_style = line_styles[counter % len(line_styles)]
    marker = markers[counter % len(markers)]
    
    plt.plot(labels, values, color=color, linestyle=line_style, marker=marker, label=model_name)
    counter += 1

# Adding title and labels
plt.title('Comparison of Different Models')
plt.xlabel('Data Points')
plt.ylabel('Values')

# Adding a legend to differentiate the lines from each model
plt.legend(title='Model', fontsize=15)
# plt.legend(title='Model', fontsize=15, loc=(1.01,0.7))

# Display the plot
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figs/code-7B.png')

# plt.show()