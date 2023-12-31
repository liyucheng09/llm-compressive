# data_processor.py is to prepare data chunks to compress
import datasets
import numpy as np

class BaseProcessor:
    def __init__(self, name, modality, load_path, cache_path, tokenizer, total_size=2**24, chunk_size=2**11):
        # total_size: the full size of the data to be compressed
        # chunk_size: the size of each chunk.
        #             default 2048, according to most LLMs' context size. 
        #             But it will change according to the model.
        #             if modal is 'text', chunk_size is the number of tokens.
        #             if modal is 'image', chunk_size is the number of pixels (bytes as in gray scale).
        #             if modal is 'audio', chunk_size is the number of bytes.

        self.name = name
        self.modality = modality
        self.load_path = load_path
        self.cache_path = cache_path
        self.tokenizer = tokenizer

        self.total_size = total_size

        self.chunk_size = chunk_size
        if self.modality == 'image':
            self.sample_patch_size = (64, 128)
        elif self.modality == 'text':
            self.sample_chunk_size = 2**14
        elif self.modality == 'audio':
            self.sample_chunk_size = 2**13

    def batches(self, batch_size):
        for i in range(0, len(self.chunks), batch_size):
            yield self.chunks[i:i+batch_size]

class MultiModalProcessor(BaseProcessor):
    def __init__(self, name, modality, load_path, cache_path, tokenizer, total_size=2**24, chunk_size=2**11, config=None):
        super().__init__(name, modality, load_path, cache_path, tokenizer, total_size, chunk_size)

        self.config = config

        self._load_dataset()
        self._data_stream()

    def tokenize(self, byte_stream):
        # huggingface tokenizer only take string as input, so here we build a byte-ids mapping and make a byte tokenizer
        # this function returns a 1D list consists of token ids
        if getattr(self, 'byte2ids', None) is None:
            byte2ids = np.zeros(256, dtype=np.int32)

            for byte in range(256):
                byte_token = f'<0x{byte:02x}>' # e.g. <0x00>
                byte2ids[byte] = self.tokenizer.convert_tokens_to_ids(byte_token)

            self.byte2ids = byte2ids
        
        byte_array = np.frombuffer(byte_stream, dtype=np.uint8)
        ids = self.byte2ids[byte_array].tolist()
        return ids

    def prepare_batches(self, context_size):
        input_ids = self.tokenize(self.stream)

        # now we chunk the long input_ids into batches
        chunks = [input_ids[i:i+context_size] for i in range(0, len(input_ids), context_size)]
        last_chunk = chunks[-1]
        chunks = chunks[:-1]

        self.bytes_droped = len(last_chunk)
        self.chunks = chunks

        self.context_size = context_size
        self._metadata()
    
    def _metadata(self):
        num_chunks = len(self.chunks)
        num_bytes = self.context_size * num_chunks
        self.metadata = {
            'name': self.name,
            'modality': self.modality,
            'load_path': self.load_path,
            'cache_path': self.cache_path,
            'total_size': self.total_size,
            'context_size': self.context_size,
            'num_chunks': num_chunks,
            'num_bytes': num_bytes,
        }
    
    def _data_stream(self):
        self.stream = self.stream[:self.total_size]

class TextProcessor(BaseProcessor):
    def __init__(self, name, modality, load_path, cache_path, total_size=2**24, chunk_size=2**11, config=None):
        super().__init__(name, modality, load_path, cache_path, total_size, chunk_size)

        self.config = config
        self._load_dataset()
        self._data_stream()

    def _data_stream(self):
        data_stream = self.all_text.encode('utf-8')[:self.total_size]
        self.all_text = data_stream.decode('utf-8', errors='ignore')
        self.stream = self.all_text.encode('utf-8')

    def prepare_batches(self, context_size):
        input_ids = self.tokenizer(self.all_text, add_special_tokens=False)['input_ids']

        # now we chunk the long input_ids into batches
        chunks = [input_ids[i:i+context_size] for i in range(0, len(input_ids), context_size)]
        last_chunk = chunks[-1]
        chunks = chunks[:-1]

        self.text_droped = self.tokenizer.decode(last_chunk)
        self.chunks = chunks

        self.context_size = context_size
        self._metadata()
    
    def _metadata(self):
        num_chunks = len(self.chunks)
        num_tokens = num_chunks * self.context_size
        num_chars = len(self.all_text) - len(self.text_droped)
        num_bytes = len(self.stream) - len(self.text_droped.encode('utf-8'))
        self.metadata = {
            'name': self.name,
            'modality': self.modality,
            'load_path': self.load_path,
            'cache_path': self.cache_path,
            'total_size': self.total_size,
            'context_size': self.context_size,
            'num_chunks': num_chunks,
            'num_tokens': num_tokens,
            'num_chars': num_chars,
            'num_bytes': num_bytes,
        }

class BBCNewsProcessor(TextProcessor):
    
    def _load_dataset(self):
        ds = datasets.load_dataset(self.load_path, self.config, split='train')
        all_text = ''
        for article in ds:
            title = article['title']
            time = article['published_date']
            text = article['content']
            line = f'BBC News {time}\nTitle: {title}\n{text}\n\n'
            all_text += line
        self.all_text = all_text

class BBCImageProcessor(MultiModalProcessor):

    def _load_dataset(self):
        ds = datasets.load_dataset(self.load_path, self.config, split='train')
        all_pixels = []
        for image in ds:
            image = image['image']
            patch = self._sample_patch(image, self.sample_patch_size)
            all_pixels += patch

        self.stream = np.array(all_pixels, dtype=np.uint8).tobytes()
    
    def _sample_patch(self, image, patch_size):
        # image: PIL image
        # patch_size: (height, width). default (64, 128)
        width, height = image.size

        if width < patch_size[1] and height < patch_size[0]:
            top, left = 0, 0
            patch_size = (height, width)
        elif width < patch_size[1]:
            top = np.random.randint(0, height - patch_size[0])
            left = 0
        elif height < patch_size[0]:
            top = 0
            left = np.random.randint(0, width - patch_size[1])
        else:
            top = np.random.randint(0, height - patch_size[0])
            left = np.random.randint(0, width - patch_size[1])

        patch = image.crop((left, top, left+patch_size[1], top+patch_size[0]))
        patch_gray = patch.convert('L')
        patch_array = np.array(patch_gray).flatten().tolist()

        return patch_array