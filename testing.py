import schlieren

#bos = schlieren.BOS()
#bos.read('PIV Challange')
#bos.compute(start=0, stop=5)
#bos.display(schlieren.DATA_COMPUTED)

import numpy as np
from compute_tools import *

a = np.arange(25).reshape(1, 5, 5)
a = np.vstack([a, a+1, a+2, a+3, a+4])
print(a.shape)

b = np.array([[[0, 1], [5, 6]]])
b = np.vstack([b] * 5)
print(b.shape)

c = batch_correlate(a, b, 'valid').astype(np.int32)

print(c.shape)
print(a)
print(b)
print('################')
print(c)