import schlieren

#bos = schlieren.BOS()
#bos.read('PIV Challange')
#bos.compute(start=0, stop=5)
#bos.display(schlieren.DATA_COMPUTED)

import time
import numpy as np
from compute_tools import *

n = 30
w = 64
h = 64
kw = 32
kh = 32

# cross pattern
pat = np.array([[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]])

# noise function
def noise(shape:tuple, scale:float=0.2) -> np.ndarray:
    # create random numbers
    out = np.random.random(shape).astype(np.float16)

    # center and scale
    out = (2 * out - 1) * scale

    # output
    return out

# create kernal
k = np.zeros((kh, kw))

# center pattern
ph, pw = pat.shape
row = (kh - ph) // 2
col = (kw - pw) // 2
k[row:row+ph, col:col+pw] = pat
print('##### K #####')
print(k.shape)

# create images
t0 = time.time()
a = noise((n, h, w)) # noise
for i in range(n):
    a[i, i:i+k.shape[0], i:i+k.shape[1]] += k

print('##### A #####')
print(a.shape)

# create kernals
b = k.reshape((1, *k.shape))
b = np.vstack([b] * n).astype(np.float16)
b += noise((n, *k.shape)) # noise

print('##### B #####')
print(b.shape)

# corrilate
t1 = time.time()
c = batch_correlate(a, b, 'valid')
print('##### C #####')
print(c.shape)

# find
t2 = time.time()
x, y = sub_pixel_peak(c)
print('##### X Y #####')
print(x, y)

t3 = time.time()

print(f'Creation time: {1000 * (t2 - t1)} ms')
print(f'Corrilation time: {1000 * (t2 - t1)} ms')
print(f'Search time: {1000 * (t3 - t2)} ms')