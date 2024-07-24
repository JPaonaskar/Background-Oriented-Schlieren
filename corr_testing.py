'''
CORR TESTING
by Josha Paonaskar

Correlation testing to validate batch methods

Resources:

'''

import time
import numpy as np
import vectorized_tools as vt

n = 1000
w = 32
h = 32
kw = 16
kh = 16

vt.get_devices(verbose=True)

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

for i in range(n):
    j = 0
    a[i, j:j+k.shape[0], j:j+k.shape[1]] += k

print('##### A #####')
print(a.shape)

# create kernals
b = k.reshape((1, *k.shape))
b = np.vstack([b] * n).astype(np.float16)
b += noise((n, *k.shape)) # noise

print('##### B #####')
print(b.shape)

# batched correlation
t1 = time.time()
print('##### C - batched #####')
c_batch = vt.normxcorr2(a, b, 'valid')
print(c_batch.shape)

# find
t2 = time.time()
print('##### C - torch #####')
c_torch = vt.normxcorr2_accel(a, b)
print(c_torch.shape)

# convert to displacements
t3 = time.time()
u, v = vt.displacement(c_batch)
print('##### U V #####')
print(u.shape, v.shape)

t4 = time.time()

print(f'Creation time: {1000.0 * (t2 - t1)} ms')
print(f'NormXCorr2 time: {1000.0 * (t2 - t1)} ms')
print(f'Torch time: {1000.0 * (t3 - t2)} ms')
print(f'Search time disp: {1000.0 * (t4 - t3)} ms')

error = np.mean(np.square(c_torch - c_batch))
print(f'MSE: {error}')