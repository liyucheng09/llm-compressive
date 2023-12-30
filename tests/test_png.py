import png
import os

text = ' '.join(['pytorch'] * 1000)
byte_stream = text.encode('utf-8')
print(len(byte_stream))

w = png.Writer(len(text), 1, greyscale=True, bitdepth=8)
with open('compressed.png', 'wb') as f:
    w.write(f, [byte_stream])
print(os.path.getsize('compressed.png'))