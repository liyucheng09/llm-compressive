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
import copy

class Metrics:

    baselines = {
        'png': png_compressor,
        'gzip': gzip_compressor,
        'zlib': zlib_compressor,
        'flac': flac_compressor
    }

    def __init__(self, modality, save_path, model_name, baselines = ['png', 'zlib', 'flac'], byte2id = None, use_arithmetic_coding = True):
        self.baselines = {baseline: Metrics.baselines[baseline] for baseline in baselines}
        self.metrics = {
            'bpb': Metrics._bpb,
            'original_size': Metrics._original_size,
            'ratio': Metrics._ratio,
            'compressed_size': Metrics._compressed_size,
            'context_size': Metrics._context_size,
            'stride': Metrics._stride,
            'batches': Metrics._num_chunks
        }
        self.modality = modality
        if modality == 'text':
            self.metrics['bpt'] = Metrics._bpt
            self.metrics['bpc'] = Metrics._bpc
        else:
            assert byte2id is not None
            self.byte2id = torch.tensor(byte2id, dtype=torch.long)
        
        self.save_path = save_path
        self.use_arithmetic_coding = use_arithmetic_coding

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
                baseline_metadata = copy.deepcopy(metadata)
                baseline_metadata['context_size'] = 'None'
                
                baseline_metrics = self._compute_metrics(compressed_size, baseline_metadata)
                with open(baseline_result_path, 'w') as f:
                    json.dump(baseline_metrics, f, ensure_ascii=False, indent=2)
        
        # Now compute metrics for the model
        model_result_path = os.path.join(save_dir, model_name + '.json')
        if self.use_arithmetic_coding:
            model_compressed_path = os.path.join(save_dir, model_name + '.compressed')
            with open(model_compressed_path, 'wb') as f:
                f.write(self.arithmetic_coding_cache)
        
        compressed_size = self.self_info_cache / 8
        model_metrics = self._compute_metrics(compressed_size, metadata)

        if self.use_arithmetic_coding:
            compressed_size = len(self.arithmetic_coding_cache)
            ac_metrics = self._compute_metrics(compressed_size, metadata)
            ac_metrics = {'ac_' + metric: ac_metrics[metric] for metric in ac_metrics}
            model_metrics.update(ac_metrics)

        with open(model_result_path, 'w') as f:
            json.dump(model_metrics, f, ensure_ascii=False, indent=2)
        
        print('Metrics computed for model {} on dataset {}'.format(model_name, name))

    def _cache_arithmetic_coding(self, pmf, sym, stride = None):
        # Due to pmf is extremely memory consuming, we thus do the 
        # cache the arithmetic coding result
        if getattr(self, 'arithmetic_coding_cache', None) is None or self.arithmetic_coding_cache == b'':
            self.arithmetic_coding_cache = b''
        elif stride is not None:
            # if stride is not None and the cache is not empty
            # then we only need to use pmf[:, -stride:, :]
            pmf = pmf[:, -stride:, :]
            sym = sym[:, -stride:]
        self.arithmetic_coding_cache += arithmetic_coding(pmf[:, :-1, :], sym[:, 1:])
    
    def _cache_self_info(self, pmf, sym, stride = None):
        if getattr(self, 'self_info_cache', None) is None or self.self_info_cache == 0:
            self.self_info_cache = 0
        elif stride is not None:
            pmf = pmf[:, -stride:, :]
            sym = sym[:, -stride:]
        self.self_info_cache += -torch.log2(pmf[:, :-1, :]).gather(dim=-1, index=sym[:, 1:].unsqueeze(-1)).sum().item()
    
    def clear_cache(self):
        self.arithmetic_coding_cache = b''
        self.self_info_cache = 0
    
    def step(self, logits, sym, stride = None):
        # if logits in dtype torch.float16, torch.bfloat16, then is it very important to convert it to torch.float32
        if logits.dtype in [torch.float16, torch.bfloat16]:
            logits = logits.to(torch.float32)
        if self.modality != 'text':
            if self.byte2id.device != logits.device:
                self.byte2id = self.byte2id.to(logits.device)

            # we restrict the output space in the byte space
            true_logits = logits.index_select(dim=-1, index=self.byte2id)
            pmf = torch.softmax(true_logits, dim=-1)

            # map the byte symbol to the index in the pure byte space coordinating the true_logits
            _, _, new_sym = torch.nonzero(self.byte2id == sym.unsqueeze(-1), as_tuple=True)
            sym=new_sym.view(sym.shape)
        else:
            pmf = torch.softmax(logits, dim=-1)

        if self.use_arithmetic_coding:
            self._cache_arithmetic_coding(pmf, sym.to(torch.int32), stride = stride)
        self._cache_self_info(pmf, sym, stride = stride)

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
    def _stride(compressed_size, metadata):
        return metadata['stride']

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

    @staticmethod
    def _context_size(compressed_size, metadata):
        # context size in the compression
        return metadata['context_size']
    
    @staticmethod
    def _num_chunks(compressed_size, metadata):
        # number of chunks
        return metadata['num_chunks']

    def _compute_metrics(self, compressed_size, metadata):
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = self.metrics[metric](compressed_size, metadata)
        return metrics