from compressor import (
    arithmetic_coding,
    png_compressor,
    gzip_compressor,
    zlib_compressor,
    flac_compressor
)

class Metrics:

    baselines = {
        'png': png_compressor,
        'gzip': gzip_compressor,
        'zlib': zlib_compressor,
        'flac': flac_compressor
    }

    def __init__(self, modality, baselines):
        if modality == 'text':
            metrics = 

    def __call__(self, ):
        return self.metrics(*args, **kwargs)