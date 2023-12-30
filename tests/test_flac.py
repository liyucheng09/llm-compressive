import numpy as np
import soundfile as sf
import os

text = ' '.join(['pytorch'] * 1000)
byte_stream = text.encode('utf-8')
print(len(byte_stream))

if len(byte_stream) % 2 != 0:
    byte_stream += b'\x00'

pesudo_audio = np.frombuffer(byte_stream, dtype=np.int16)
print(pesudo_audio.shape)

sample_rate = 16000

sf.write('pesudo_audio.flac', pesudo_audio, sample_rate)
print(os.path.getsize('pesudo_audio.flac'))