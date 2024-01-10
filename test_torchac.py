from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import torch
import torchac_ as torchac

def pmf_to_cdf(pmf):
  cdf = pmf.cumsum(dim=-1)
  spatial_dimensions = pmf.shape[:-1] + (1,)
  zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
  cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
  # On GPU, softmax followed by cumsum can lead to the final value being 
  # slightly bigger than 1, so we clamp.
  cdf_with_0 = cdf_with_0.clamp(max=1.)
  return cdf_with_0

model = AutoModelForCausalLM.from_pretrained("/mnt/fast/nobackup/scratch4weeks/yl02706/models/Mistral-7B", device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/mnt/fast/nobackup/scratch4weeks/yl02706/models/Mistral-7B", use_fast=False, trust_remote_code=True)

text = ' '.join(['pytorch'] * 500)

inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
outputs = model(**inputs)

print(outputs.logits.dtype)

probs = outputs.logits.softmax(dim=-1)
print(probs.shape)

cdf = pmf_to_cdf(probs)

bits = -torch.log2(probs).gather(dim=-1, index=inputs['input_ids'].unsqueeze(-1)).squeeze(-1).sum()
print(bits)
print(bits/8)
print(text.encode('utf-8').__len__())

cdf = cdf.detach().cpu()

sym_32 = inputs['input_ids'].to(torch.int32).detach().cpu()
sym_16 = inputs['input_ids'].to(torch.int16).detach().cpu()

byte_stream_32 = torchac.encode_float_cdf(cdf, sym_32)
print(len(byte_stream_32))

# byte_stream_16 = torchac.encode_float_cdf(cdf, sym_16, precision=16)
# print(len(byte_stream_16))

d = torchac.decode_float_cdf(cdf, byte_stream_32)
# print(len(byte_stream_32))
print('=========================')

assert sym_32.equal(d)

# torchac.decode_float_cdf(cdf, byte_stream, sym)

# # print(torchac.decode_float_cdf(cdf, byte_stream).shape)
# # print(inputs['input_ids'].shape)