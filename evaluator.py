from compressor import (
    arithmetic_coding,
    png_compressor,
    gzip_compressor,
    zlib_compressor,
    flac_compressor
)
import os
import json

class Metrics:

    baselines = {
        'png': png_compressor,
        'gzip': gzip_compressor,
        'zlib': zlib_compressor,
        'flac': flac_compressor
    }

    def __init__(self, modality, save_path, baselines = ['png', 'zlib', 'flac']):
        self.baselines = {baseline: Metrics.baselines[baseline] for baseline in baselines}
        self.metrics = {
            'bpb': Metrics._bpb,
            'ratio': Metrics._ratio
        }
        if modality == 'text':
            self.metrics['bpt'] = Metrics._bpt
            self.metrics['bpc'] = Metrics._bpc
        self.save_path = save_path

    def __call__(self, data_stream, metadata, model_name, pmf = None, sym = None):
        # we will create a file for every model, under the save_dir.
        name = metadata['name']
        save_dir = os.path.join(self.save_path, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # check whether baselines have been computed before
        for baseline in self.baselines:
            baseline_result_path = os.path.join(save_dir, baseline + '.json')
            baseline_compressed_path = os.path.join(save_dir, baseline + '.compressed')
            if not os.path.exists(baseline_result_path):
                # compute baseline
                compressed_size = self.baselines[baseline](data_stream, baseline_compressed_path)
                baseline_metrics = self._compute_metrics(compressed_size, metadata)
                with open(baseline_result_path, 'w') as f:
                    json.dump(baseline_metrics, f, ensure_ascii=False, indent=2)
        
        # Now compute metrics for the model
        model_result_path = os.path.join(save_dir, model_name + '.json')
        model_compressed_path = os.path.join(save_dir, model_name + '.compressed')
        compressed_size = arithmetic_coding(pmf, sym, model_compressed_path)
        model_metrics = self._compute_metrics(compressed_size, metadata)
        with open(model_result_path, 'w') as f:
            json.dump(model_metrics, f, ensure_ascii=False, indent=2)

        print('Metrics computed for model {} on dataset {}'.format(model_name, name))

    @staticmethod
    def _bpb(compressed_size, metadata):
        # bits per byte
        num_bytes = metadata['num_bytes']
        return compressed_size * 8 / num_bytes
    
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