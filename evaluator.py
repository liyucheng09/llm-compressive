from compressor import (
    arithmetic_coding,
    png_compressor,
    gzip_compressor,
    zlib_compressor,
    flac_compressor
)
import os
import json
import torch

class Metrics:

    baselines = {
        'png': png_compressor,
        'gzip': gzip_compressor,
        'zlib': zlib_compressor,
        'flac': flac_compressor
    }

    def __init__(self, modality, save_path, model_name, baselines = ['png', 'zlib', 'flac']):
        self.baselines = {baseline: Metrics.baselines[baseline] for baseline in baselines}
        self.metrics = {
            'bpb': Metrics._bpb,
            'original_size': Metrics._original_size,
            'ratio': Metrics._ratio,
            'compressed_size': Metrics._compressed_size,
        }
        if modality == 'text':
            self.metrics['bpt'] = Metrics._bpt
            self.metrics['bpc'] = Metrics._bpc
        
        self.save_path = save_path
        self.use_arithmetic_coding = True if ('llama' in model_name.lower() or 'mistral' in model_name.lower()) else False

    def __call__(self, data_stream, metadata, model_name):
        # we will create a file for every model, under the save_dir.
        name = metadata['name']
        time = metadata['time']
        save_dir = os.path.join(self.save_path, name, time)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # check whether baselines have been computed before
        for baseline in self.baselines:
            baseline_result_path = os.path.join(save_dir, baseline + '.json')
            baseline_compressed_path = os.path.join(save_dir, baseline + '.compressed')
            if not os.path.exists(baseline_result_path):
                # compute baseline
                compressed_size = self.baselines[baseline](data_stream, save_path = baseline_compressed_path)
                baseline_metrics = self._compute_metrics(compressed_size, metadata)
                with open(baseline_result_path, 'w') as f:
                    json.dump(baseline_metrics, f, ensure_ascii=False, indent=2)
        
        # Now compute metrics for the model
        model_result_path = os.path.join(save_dir, model_name + '.json')
        if self.use_arithmetic_coding:
            model_compressed_path = os.path.join(save_dir, model_name + '.compressed')
            with open(model_compressed_path, 'wb') as f:
                f.write(self.arithmetic_coding_cache)
        
        compressed_size = self.self_info_cache.item() / 8
        model_metrics = self._compute_metrics(compressed_size, metadata)

        if self.use_arithmetic_coding:
            compressed_size = len(self.arithmetic_coding_cache)
            ac_metrics = self._compute_metrics(compressed_size, metadata)
            ac_metrics = {'ac_' + metric: ac_metrics[metric] for metric in ac_metrics}
            model_metrics.update(ac_metrics)

        with open(model_result_path, 'w') as f:
            json.dump(model_metrics, f, ensure_ascii=False, indent=2)
        
        print('Metrics computed for model {} on dataset {}'.format(model_name, name))

    def _cache_arithmetic_coding(self, pmf, sym):
        # Due to pmf is extremely memory consuming, we thus do the 
        # cache the arithmetic coding result
        if getattr(self, 'arithmetic_coding_cache', None) is None:
            self.arithmetic_coding_cache = b''
        self.arithmetic_coding_cache += arithmetic_coding(pmf, sym)
    
    def _cache_self_info(self, pmf, sym):
        if getattr(self, 'self_info_cache', None) is None:
            self.self_info_cache = 0
        self.self_info_cache += -torch.log2(pmf[:, :-1, :]).gather(dim=-1, index=sym[:, 1:].unsqueeze(-1)).squeeze(-1).sum()
    
    def clear_cache(self):
        self.arithmetic_coding_cache = b''
        self.self_info_cache = 0
    
    def step(self, pmf, sym):
        if self.use_arithmetic_coding:
            self._cache_arithmetic_coding(pmf, sym.to(torch.int16))
        self._cache_self_info(pmf, sym)

    @staticmethod
    def _bpb(compressed_size, metadata):
        # bits per byte
        num_bytes = metadata['num_bytes']
        return compressed_size * 8 / num_bytes

    @staticmethod
    def _original_size(compressed_size, metadata):
        # bits per byte
        num_bytes = metadata['num_bytes']
        return num_bytes

    @staticmethod
    def _compressed_size(compressed_size, metadata):
        # bits per byte
        return compressed_size
    
    @staticmethod
    def _bpt(compressed_size, metadata):
        # bits per token
        num_tokens = metadata['num_tokens']
        return compressed_size * 8 / num_tokens
    
    @staticmethod
    def _bpc(compressed_size, metadata):
        # bits per character
        num_chars = metadata['num_chars']
        return compressed_size * 8 / num_chars
    
    @staticmethod
    def _ratio(compressed_size, metadata):
        # compression ratio
        num_bytes = metadata['num_bytes']
        return compressed_size / num_bytes

    def _compute_metrics(self, compressed_size, metadata):
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = self.metrics[metric](compressed_size, metadata)
        return metrics