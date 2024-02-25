import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline as pyo
import plotly.io as pio

base_model_name_to_label = {
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
    'LLaMA-65B': 'LLaMA-65B',
    'Yi-34B-200K': 'Yi-34B',
    'Qwen1.5-7B': 'Qwen1.5-7B',
    'gemma-7B': 'Gemma-7B',
}

long_context_models = [
    'Baichuan2'
    'Mistral',
    'Llama',
    'LLaMA',
    'Yi',
    'chatglm3',
    'gemma',
    'Qwen',
    'Qwen1.5',
]

long_context_unselected_defaultly = [
    'LLaMA',
]

unselected_defaultly = [
    'Internlm-7B',
    'Chatglm3-6B',
    'LLaMA-7B',
    'LLaMA-13B',
    'Llama-2-13B',
    'Yi-34B',
    'LLaMA-65B',
    'CodeLlama-7B',
    'Qwen-7B',
    'Llama-2-70B',
]

if __name__ == "__main__":
    # tasks = ['wikitext']
    tasks = ['wikitext', 'arxiv', 'bbc_news', 'code', 'math', 'bbc_image']
    line_styles = ['solid', 'dot', 'dash', 'longdash', 'dashdot']
    markers = ['circle', 'square', 'diamond', 'cross', 'x']

    df_for_tasks = {}
    for task in tasks:
        df = pd.read_json(f'results/{task}_results.json')
        df.index = df.index.strftime('%Y-%m')
        df.dropna(axis=1, how='any', inplace=True)
        df = df.applymap(lambda x: x * 100)
        df = df[[col for col in df.columns if col in base_model_name_to_label]]
        df = df.rename(columns=base_model_name_to_label)
        df_for_tasks[task] = df

    for task, df in df_for_tasks.items():
        fig = go.Figure()
        for i, model in enumerate(df.columns):
            visible = "legendonly" if model in unselected_defaultly else True
            if 'codellama' in model.lower() and task in ['code', 'math', 'bbc_image']:
                visible = True
            fig.add_trace(go.Scatter(x=df.index, y=df[model], mode='lines+markers', name=model, line=dict(dash=line_styles[i%len(line_styles)]),
                                    marker=dict(symbol=markers[i%len(markers)], size=4), visible=visible))

        fig.update_layout(title=task, xaxis_title='Date', yaxis_title='Compression Ratio', xaxis_fixedrange=True, yaxis_fixedrange=True)
        pio.write_html(fig, file=f'page/{task}.html', include_plotlyjs='cdn')

    # process results/long_wikitext_results.json
    df = pd.read_json('results/long_wikitext_results.json')
    df.index = df.index.strftime('%Y-%m')
    df = df.applymap(lambda x: x * 100)
    df = df[[col for col in df.columns if any([model.lower() in col.lower() for model in long_context_models])]]

    context_sizes = []
    models = []
    avg_perform = []
    results = df.mean(axis=0)
    for model, perf in results.items():
        model, context = model.rsplit('-', 1)
        context_sizes.append(int(context[:-1]))
        models.append(model)
        avg_perform.append(perf)

    plot_df = pd.DataFrame({'Model': models, 'Context Size': context_sizes, 'Average Performance': avg_perform})
    plot_df = plot_df.sort_values(by='Context Size')
    fig = go.Figure()

    for i, model in enumerate(plot_df['Model']):
        visible = "legendonly" if model in long_context_unselected_defaultly else True
        fig.add_trace(go.Scatter(x=plot_df['Context Size'], y=plot_df['Average Performance'], mode='lines+markers', name=model, line=dict(dash=line_styles[i%len(line_styles)]),
                                marker=dict(symbol=markers[i%len(markers)], size=4), visible=visible))
    fig.update_layout(title='Wikitext', xaxis_title='Context Size', yaxis_title='Compression Ratio (%, across all times)', xaxis_fixedrange=True, yaxis_fixedrange=True)
    fig.update_xaxes(
        tickvals=context_sizes,
        ticktext=[f'{size}k' for size in context_sizes]
    )
    pio.write_html(fig, file='page/wikitext_context.html', include_plotlyjs='cdn')

    print('Done')
