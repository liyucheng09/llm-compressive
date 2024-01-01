import torch
import torchac
import png
import os
import gzip
import zlib
import numpy as np
import soundfile as sf

def arithmetic_coding(pmf, sym, save_path = None):
    # pmf is the output of language model after softmax, which is the prob distribution over the vocab
    # sym is input_ids
    def pmf_to_cdf(pmf):
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        # On GPU, softmax followed by cumsum can lead to the final value being 
        # slightly bigger than 1, so we clamp.
        cdf_with_0 = cdf_with_0.clamp(max=1.)
        return cdf_with_0
    
    cdf = pmf_to_cdf(pmf)
    if cdf.device.type != 'cpu':
        cdf = cdf.detach().cpu()
    if sym.device.type != 'cpu':
        sym = sym.detach().cpu()
    
    byte_stream = torchac.encode_float_cdf(cdf, sym)
    if save_path is not None:
        with open(save_path, 'wb') as f:
            f.write(byte_stream)
    
    return byte_stream

def png_compressor(byte_stream, save_path = None):
    w = png.Writer(len(byte_stream), 1, greyscale=True, bitdepth=8)
    if save_path is None:
        save_path = 'compressed.png'
    with open(save_path, 'wb') as f:
        w.write(f, [byte_stream])
    compressed_size = os.path.getsize(save_path)
    return compressed_size

def gzip_compressor(byte_stream, save_path = None):
    if save_path is None:
        save_path = 'compressed.gz'
    with gzip.open(save_path, 'wb') as f:
        f.write(byte_stream)
    compressed_size = os.path.getsize(save_path)
    return compressed_size

def zlib_compressor(byte_stream, wbits = 15, save_path = None):
    compressor = zlib.compressobj(level=9, wbits=wbits)
    compressed = compressor.compress(byte_stream) + compressor.flush()
    if save_path is not None:
        with open(save_path, 'wb') as f:
            f.write(compressed)
    compressed_size = len(compressed)
    return compressed_size

def flac_compressor(byte_stream, save_path = None):
    if len(byte_stream) % 2 != 0:
        byte_stream = byte_stream + b'\x00'
    pesudo_audio = np.frombuffer(byte_stream, dtype=np.int16)
    sample_rate = 16000
    if save_path is None:
        save_path = 'compressed.flac'
    else:
        save_path = save_path + '.flac'
    sf.write(save_path, pesudo_audio, sample_rate)
    compressed_size = os.path.getsize(save_path)
    return compressed_size
