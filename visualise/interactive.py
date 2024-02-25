import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline as pyo

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
    'LLaMA-30B': 'LLaMA-30B',
    'LLaMA-65B': 'LLaMA-65B',
    'Yi-34B-200K': 'Yi-34B',
    'Qwen1.5-7B': 'Qwen1.5-7B',
    'gemma-7B': 'Gemma-7B',
}

unselected_defaultly = [
    'Internlm-7B',
    'Chatglm3-6B',
    'LLaMA-7B',
    'LLaMA-13B',
    'Llama-2-13B',
    'Yi-34B',
    'LLaMA-30B',
    'LLaMA-65B',
    'CodeLlama-7B',
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
        df = df.applymap(lambda x: x['ratio'] * 100)
        df = df[[col for col in df.columns if col in base_model_name_to_label]]
        df = df.rename(columns=base_model_name_to_label)
        df_for_tasks[task] = df

    for task, df in df_for_tasks.items():
        fig = make_subplots(rows=1, cols=1, shared_yaxes=True, subplot_titles=['Wikitext'])
        for i, model in enumerate(df.columns):
            visible = "legendonly" if model in unselected_defaultly else True
            fig.add_trace(go.Scatter(x=df.index, y=df[model], mode='lines+markers', name=model, line=dict(dash=line_styles[i%len(line_styles)]),
                                    marker=dict(symbol=markers[i%len(markers)], size=4), visible=visible), row=1, col=1)

        fig.update_layout(title=task, xaxis_title='Date', yaxis_title='Compression Ratio', xaxis_fixedrange=True, yaxis_fixedrange=True)
        pyo.plot(fig, filename=f'page/{task}.html', include_plotlyjs='cdn')
