'''
CORR TESTING
by Josha Paonaskar

Correlation testing to validate batch methods

Resources:

'''

import time
import numpy as np
from batch_tools import *

n = 4
w = 7
h = 7
kw = 3
kh = 3

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
a = noise((n, h, w), scale=0) # noise
i0 = min(w - kw, h - kh) - n
print(i0)
if (i0 < 0): raise ValueError(f'i0 ({i0}) is out of bounds. Decrease n or kernal size or increase image size')
for i in range(n):
    j = i0 + i
    a[i, j:j+k.shape[0], j:j+k.shape[1]] += k

print('##### A #####')
print(a)

# create kernals
b = k.reshape((1, *k.shape))
b = np.vstack([b] * n).astype(np.float16)
#b += noise((n, *k.shape)) # noise

print('##### B #####')
print(b)

# corrilate
t1 = time.time()
c = normxcorr2(a, b, 'valid')
print('##### C #####')
print(c)

# find
t2 = time.time()
x, y = peak(c)
print('##### X Y #####')
print(x, y)

# convert to displacements
t3 = time.time()
u, v = displacement(c)
print('##### U V #####')
print(u, v)

t4 = time.time()

print(f'Creation time: {1000 * (t2 - t1)} ms')
print(f'Correlation time: {1000 * (t2 - t1)} ms')
print(f'Search time peak: {1000 * (t3 - t2)} ms')
print(f'Search time disp: {1000 * (t4 - t3)} ms')