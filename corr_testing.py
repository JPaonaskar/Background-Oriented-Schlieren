'''
CORR TESTING
by Josha Paonaskar

Correlation testing to validate batch methods

Resources:

'''

import time
import numpy as np
import vectorized_tools as bt
from normxcorr2 import normxcorr2

n = 16
w = 32
h = 32
kw = 16
kh = 16

# cross pattern
pat = np.array([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]])

# noise function
def noise(shape:tuple, scale:float=0.1) -> np.ndarray:
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
print(k)

# create images
t0 = time.time()
a = noise((n, h, w)) # noise
i0 = min(w - kw, h - kh) - n 

if (i0 < 0): raise ValueError(f'i0 ({i0}) is out of bounds. Decrease n or kernal size or increase image size')
for i in range(n):
    j = i0 + i
    a[i, j:j+k.shape[0], j:j+k.shape[1]] += k

print('##### A #####')
print(a.shape)

# create kernals
b = k.reshape((1, *k.shape))
b = np.vstack([b] * n).astype(np.float16)
b += noise((n, *k.shape)) # noise

print('##### B #####')
print(b.shape)

# looped correlation
t1 = time.time()
c_norm = []
for i in range(len(a)):
    c_norm.append(normxcorr2(b[i], a[i], 'full'))
c_norm = np.array(c_norm)
print('##### C - normxcorr2 #####')
print(c_norm.shape)

# batched correlation
t2 = time.time()
c_batch = bt.normxcorr2(a, b, 'full')
print('##### C - batched #####')
print(c_batch.shape)

# find
t3 = time.time()
x, y = bt.peak(c_batch)
print('##### X Y #####')
print(x, y)

# convert to displacements
t4 = time.time()
u, v = bt.displacement(c_batch)
print('##### U V #####')
print(u, v)

t5 = time.time()

print(f'Creation time: {1000.0 * (t2 - t1)} ms')
print(f'NormXCorr2 time: {1000.0 * (t2 - t1)} ms')
print(f'Batched time: {1000.0 * (t3 - t2)} ms')
print(f'Search time peak: {1000.0 * (t4 - t3)} ms')
print(f'Search time disp: {1000.0 * (t5 - t4)} ms')

error = np.mean(np.square(c_norm - c_batch))
print(f'MSE: {error}')