FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /workspace/
RUN pip install --upgrade pip \
    && pip install vllm transformers datasets accelerate nltk flash_attn --no-cache-dir \
    && pip uninstall transformer-engine -y

RUN pip install evaluate rouge rouge_score --no-cache-dir