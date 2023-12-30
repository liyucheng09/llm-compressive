import gzip
import os

text = ' '.join(['pytorch'] * 1000)
byte_stream = text.encode('utf-8')
print(len(byte_stream))

with gzip.open('compressed.gz', 'wb') as f:
    f.write(byte_stream)

print(os.path.getsize('compressed.gz'))