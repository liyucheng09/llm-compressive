import zlib

text = ' '.join(['pytorch'] * 1000)
byte_stream = text.encode('utf-8')
print(len(byte_stream))

# level=9, wbits=15 is the same as gzip 
compressor = zlib.compressobj(level=9, wbits=15)
compressed = compressor.compress(byte_stream) + compressor.flush()
print(len(compressed))