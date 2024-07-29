'''
BENCHMARK
by Josha Paonaskar

Benchmarking to analize effeciency

Resources:

'''

import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import vectorized_tools as vt

PATTERN = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

def noise(shape:tuple, scale:float=0.1) -> np.ndarray:
    '''
    Create a random noise sample

    Args:
        shape (tuple) : noise shape
        scale (float) : amplitude of the noise (default=0.1)

    Returns:
        out (np.ndarray) : noise sample
    '''
    # create random numbers
    out = np.random.random(shape).astype(np.float16)

    # center and scale
    out = (2 * out - 1) * scale

    # output
    return out

def synthetic_dataset(batch_size:int, win_size:int=32, search_size:int=64, noise_scale:float=0.1) -> np.ndarray:
    '''
    Create a synthetic dataset for testing and banchmarking

    Args:
        batch_size (int) : size of a batch of data
        win_size (int) : window size
        search_size (int) : search size
        noise_scale (float) : noise amplitude

    Returns:
        windows (np.ndarray) : windows
        searches (np.ndarray) : search areas
        u (np.ndarray) : x displacements
        v (np.ndarray) : y displacements
    '''
    h, w = PATTERN.shape

    # pattern coordinates
    x = np.random.randint(win_size - w + 1, size=(batch_size))
    y = np.random.randint(win_size - h + 1, size=(batch_size))

    # window coordinates
    s = search_size - win_size + 1
    wx = np.random.randint(s, size=(batch_size))
    wy = np.random.randint(s, size=(batch_size))

    # build window and search
    windows = np.zeros((batch_size, win_size, win_size))
    searches = np.zeros((batch_size, search_size, search_size))

    # place pattern and windows
    for i in range(batch_size):
        windows[i, y[i]:y[i]+h, x[i]:x[i]+w] = PATTERN
        searches[i, wy[i]:wy[i]+win_size, wx[i]:wx[i]+win_size] = windows[i]

    # add noise
    windows += noise((batch_size, win_size, win_size), scale=noise_scale)
    searches += noise((batch_size, search_size, search_size), scale=noise_scale)

    # calculate displacements
    u = wx - s * 0.5 + 0.5
    v = wy - s * 0.5 + 0.5

    # return results
    return windows, searches, u, v

def batch_size_test(batch_sizes:list[int], test_window:float=5.0, win_size:int=32, search_size:int=64, noise_scale:float=0.1, show:bool=True) -> None:
    '''
    Test computation time vs batch size

    Args:
        batch_sizes (list[int]) : set of batch sizes to test
        test_window (float) : time window to test function (default=2.0)
        win_size (int) : window size (default=32)
        search_size (int) : search size (default=64)
        noise_scale (float) : noise amplitude (default=0.1)
        show (bool) : show plot results (default=True)

    Returns:
        None
    '''
    # times
    t = []

    # test each batch size
    for batch_size in tqdm(batch_sizes):
        # create sample data
        windows, searches, u, v = synthetic_dataset(batch_size, win_size, search_size, noise_scale=noise_scale)

        # sample set to average
        t_start = time.time()
        t_end = t_start
        count = 0

        # while still in the test window
        while (t_end - t_start < test_window):
            # test
            corr = vt.normxcorr2(searches, windows)
            results = vt.displacement(corr)

            # store data
            count += 1
            t_end = time.time()

        # compute frames per second
        t.append(batch_size * count / (t_end - t_start))

    # to numpy
    t = np.array(t)
    batch_sizes = np.array(batch_sizes)

    # plot
    plt.figure()
    plt.plot(batch_sizes, t)
    plt.xlabel('Batch Size')
    plt.ylabel('Frames per second')

    # show
    if show:
        plt.show()