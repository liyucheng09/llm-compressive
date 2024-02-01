# data_processor.py is to prepare data chunks to compress
import datasets
import numpy as np
import os
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import multiprocessing
import pickle
import time
from tqdm import tqdm
import re

global_tokenizer = None
def set_global_tokenizer(tokenizer):
    global global_tokenizer
    global_tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True, trust_remote_code=True)

class BaseProcessor:
    def __init__(self, name, modality, load_path, cache_path, tokenizer, config, total_size=2**24, chunk_size=2**11):
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
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        self.config = config

        self.tokenizer = tokenizer
        self.total_size = total_size

        self.chunk_size = chunk_size
        if self.modality == 'image':
            self.sample_patch_size = (32, 64)
        elif self.modality == 'text':
            self.sample_chunk_size = 2**14
        elif self.modality == 'audio':
            self.sample_chunk_size = 2**16

        # self._check_and_load_cached_input_ids()

    def _check_and_load_cached_input_ids(self):
        cache_path = os.path.join(self.cache_path, f'{self.config}.input_ids')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.input_ids = pickle.load(f)

    def batches(self, batch_size):
        for i in range(0, len(self.chunks), batch_size):
            yield self.chunks[i:i+batch_size]
    
    def _cache_tokenized(self):
        assert getattr(self, 'input_ids', None) is not None, 'Please run _data_stream() first.'
        cache_path = os.path.join(self.cache_path, f'{self.config}.input_ids')
        with open(cache_path, 'wb') as f:
            pickle.dump(self.input_ids, f)
        print(f'Saved tokenized input_ids to {cache_path}')
    
    @staticmethod
    def _tokenize_chunk(text):
        return global_tokenizer(text, add_special_tokens=False)['input_ids']

class MultiModalProcessor(BaseProcessor):
    def __init__(self, name, modality, load_path, cache_path, tokenizer, config, total_size=2**20, chunk_size=2**11):
        super().__init__(name, modality, load_path, cache_path, tokenizer, config, total_size, chunk_size)

        self._load_dataset()
        self._prepare_tokenizer()
        self._data_stream()
    
    def _prepare_tokenizer(self):
        byte2ids = np.zeros(256, dtype=np.int32)

        for byte in range(256):
            byte_token = f'<0x{byte:02X}>' # e.g. <0x00>
            byte2ids[byte] = self.tokenizer.convert_tokens_to_ids(byte_token)

        self.byte2ids = byte2ids

    @staticmethod
    def tokenize(byte_stream, byte2ids):
        # huggingface tokenizer only take string as input, so here we build a byte-ids mapping and make a byte tokenizer
        # this function returns a 1D list consists of token ids
        
        byte_array = np.frombuffer(byte_stream, dtype=np.uint8)
        ids = byte2ids[byte_array].tolist()
        return ids

    def prepare_batches(self, context_size, stride=None):
        assert getattr(self, 'input_ids', None) is not None, 'Please run _data_stream() first.'
        input_ids = self.input_ids

        if stride is None:
            # now we chunk the long input_ids into batches
            chunks = [input_ids[i:i+context_size] for i in range(0, len(input_ids), context_size)]
            last_chunk = chunks[-1]
            chunks = chunks[:-1]
        else:
            assert stride < context_size, 'Stride should be smaller than context size.'
            self.stride = stride
            chunks = []
            for i in range(0, len(input_ids), stride):
                chunk = input_ids[i:i+context_size]
                if len(chunk) < context_size:
                    break
                chunks.append(chunk)
            last_chunk = input_ids[i + (context_size - stride):]

        self.bytes_droped = len(last_chunk)
        self.chunks = chunks

        self.context_size = context_size
        self._metadata()
    
    def _metadata(self):
        num_chunks = len(self.chunks)
        num_bytes = self.context_size * num_chunks
        stride =  getattr(self, 'stride', 'None')
        self.metadata = {
            'name': self.name,
            'modality': self.modality,
            'time': self.config,
            'load_path': self.load_path,
            'cache_path': self.cache_path,
            'total_size': self.total_size,
            'context_size': self.context_size,
            'num_chunks': num_chunks,
            'num_bytes': num_bytes,
            'stride': stride,
        }
    
    def _data_stream(self):
        print(f'Tokenizing Bytes {self.name} {self.config}...')
        self.stream = self.stream[:self.total_size]

        chunk_size = 2**13
        byte_chunks = [self.stream[i:i+chunk_size] for i in range(0, len(self.stream), chunk_size)]

        num_workers = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_workers) as pool:
            result = pool.starmap(self.tokenize, [(chunk, self.byte2ids) for chunk in byte_chunks])

        input_ids = []
        for chunk in result:
            input_ids += chunk

        self.input_ids = input_ids

class TextProcessor(BaseProcessor):
    def __init__(self, name, modality, load_path, cache_path, tokenizer, config, total_size=2**23, chunk_size=2**11, **kwargs):
        super().__init__(name, modality, load_path, cache_path, tokenizer, config, total_size, chunk_size)

        self._load_dataset(**kwargs)
        self._data_stream()

    def _data_stream(self):
        print(f'Tokenizing {self.name} {self.config}...')
        start_time = time.time()

        # For code data, sents are a list of code files
        # For non-code data, sents are a list of sentences
        sents = []
        size_counter = 0
        for sent in self.sents:
            size_counter += len(sent.encode())
            if size_counter > self.total_size:
                break
            sents.append(sent)

        # For non-code data, we tokenize in sent-level.
        # So add space between sentences here.
        if self.name != 'code':
            for i in range(1, len(sents)):
                sents[i] = ' ' + sents[i]
        
        self.all_text = ''.join(sents)
        self.stream = self.all_text.encode('utf-8')

        if self.name == 'code' or self.name == 'arxiv':
            chunk_size = 1
        else:
            chunk_size = 100
        sent_chunks = [sents[i:i+chunk_size] for i in range(0, len(sents), chunk_size)]

        num_workers = multiprocessing.cpu_count()
        tokenizer_path = self.tokenizer.name_or_path
        with multiprocessing.Pool(processes=num_workers, initializer=set_global_tokenizer, initargs=(tokenizer_path,)) as pool:
            result = pool.map(self._tokenize_chunk, sent_chunks)
        
        input_ids = []
        for chunk in result:
            for sent in chunk:
                input_ids += sent
        
        self.input_ids = input_ids

        end_time = time.time()
        print(f'Tokenization finished. Time used: {end_time - start_time:.2f}s')
        # self._cache_chunks_and_metadata()

    def prepare_batches(self, context_size, stride=None):
        assert getattr(self, 'input_ids', None) is not None, 'Please run _data_stream() first.'
        input_ids = self.input_ids

        if stride is None:
            # now we chunk the long input_ids into batches
            chunks = [input_ids[i:i+context_size] for i in range(0, len(input_ids), context_size)]
            last_chunk = chunks[-1]
            chunks = chunks[:-1]

            self.text_droped = self.tokenizer.decode(last_chunk)
        else:
            assert stride < context_size, 'Stride should be smaller than context size.'
            self.stride = stride
            chunks = []
            for i in range(0, len(input_ids), stride):
                chunk = input_ids[i:i+context_size]
                if len(chunk) < context_size:
                    break
                chunks.append(chunk)
            self.text_droped = self.tokenizer.decode(input_ids[i + (context_size - stride):])

        self.chunks = chunks
        self.context_size = context_size
        self._metadata()
    
    def _metadata(self):
        num_chunks = len(self.chunks)
        num_tokens = num_chunks * self.context_size
        num_chars = len(self.all_text) - len(self.text_droped)
        num_bytes = len(self.stream) - len(self.text_droped.encode('utf-8'))
        stride =  getattr(self, 'stride', 'None')
        self.metadata = {
            'name': self.name,
            'modality': self.modality,
            'time': self.config,
            'load_path': self.load_path,
            'cache_path': self.cache_path,
            'total_size': self.total_size,
            'context_size': self.context_size,
            'num_chunks': num_chunks,
            'num_tokens': num_tokens,
            'num_chars': num_chars,
            'num_bytes': num_bytes,
            'stride': stride,
        }

class BBCNewsProcessor(TextProcessor):
    
    def _load_dataset(self, num_articles=1000):
        ds = datasets.load_dataset(self.load_path, self.config, split='train')
        all_sents = []
        if num_articles > len(ds):
            num_articles = len(ds)
        ds = ds.select(range(num_articles))
        for article in ds:
            text = article['content']
            sents = sent_tokenize(text)
            all_sents += sents
        self.sents = all_sents

class WikiTextProcessor(TextProcessor):
    def _load_dataset(self, num_sents_per_article=80):
        ds = datasets.load_dataset(self.load_path, self.config, split='train')
        all_sents = []
        for article in ds:
            text = article['text']
            sents = sent_tokenize(text)
            all_sents += sents[:num_sents_per_article]
        self.sents = all_sents

class CodeProcessor(TextProcessor):
    def _load_dataset(self):
        ds = datasets.load_dataset(self.load_path, self.config, split='train')
        ds = ds.shuffle(seed=42)
        
        all_sents = []
        for code in ds:
            # If the code file is over 500 lines, we sample a 500 lines continues chunk from the code
            code_lines = code['code'].splitlines()
            if len(code_lines) > 500:
                start = np.random.randint(0, len(code_lines) - 500)
                code_lines = code_lines[start:start+500]
            code_ = '\n'.join(code_lines)
            all_sents.append(code_) 

        self.sents = all_sents

class ArxivProcessor(TextProcessor):
    def _load_dataset(self, num_sections = 2):
        ds = datasets.load_dataset(self.load_path, self.config, split='train')
        all_sents = []
        for article in ds:
            text = article['text']
            sections = self._process_article(text)
            if sections is None:
                continue
            all_sents += sections[:num_sections]
        self.sents = all_sents
    
    def _process_article(self, text):
        def beautify_context(context: str) -> str:
            context = context.replace("<cit.>", '').replace('<ref>', '')
            context = re.sub(r"\s+", " ", context)
            context = re.sub(r"\n+", " ", context)
            return context

        text = re.sub(r"^.*?(§)", r"\1", text, flags=re.DOTALL)
        sections = re.split(r"(?<!§\.)§\s", text)
        sections = [beautify_context(section) for section in sections if section.strip()]
        if not sections:
            return None
        return sections

class BBCImageProcessor(MultiModalProcessor):

    def _load_dataset(self):
        ds = datasets.load_dataset(self.load_path, self.config, split='train')
        all_pixels = []
        for image in ds:
            image = image['img']
            if image is None:
                continue
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

class AudioProcessor(MultiModalProcessor):
    def _load_dataset(self):
        ds = datasets.load_dataset(self.load_path, self.config, split='train')
        all_bytes = b''
        for audio in ds:
            # audio bytes stream
            audio = audio['audio']

            # sample multiple chunk of audio
            chunk = self._sample_chunk(audio, self.sample_chunk_size)
            all_bytes += chunk

        self.stream = all_bytes
    
    def _sample_chunk(self, audio, chunk_size, num_chunks_per_file = 50):
        # audio: bytes stream
        # chunk_size: number of bytes per chunk
        # num_chunks_per_file: number of chunks to sample from each audio file
        num_bytes = len(audio)
        if num_bytes < chunk_size * num_chunks_per_file:
            return audio
        
        chunks = b''
        for i in range(num_chunks_per_file):
            start = np.random.randint(0, num_bytes - chunk_size)
            chunk = audio[start:start+chunk_size]
            chunks += chunk
        
        return chunks