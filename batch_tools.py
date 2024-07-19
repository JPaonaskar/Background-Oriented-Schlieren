'''
BATCH_TOOLS
by Josha Paonaskar

Batch based methods for pixel correlation and tracking

Resources:
    http://www.learnpiv.org/subPixel/
    https://github.com/Sabrewarrior/normxcorr2-python
'''

import numpy as np
from scipy.signal import fftconvolve

def normxcorr2(images:np.ndarray, kernals:np.ndarray, mode:str='full') -> np.ndarray:
    '''
    Normalized cross correlation for batches
    Modified version of Sabrewarrior/normxcorr2-python (original code is in /normxcorr2-python) <- NOT IMPLEMENTED YET

    Args:
        kernals:np.ndarray : batch of kernal images
        images:np.ndarray : batch of images

    Returns:
        corr (np.ndarray) : batch of correlation values
    '''
    # flip kernal vertically (axis=1) and horizontally (axis=2)
    kernals = np.flip(np.flip(kernals, axis=1), axis=2)

    # get means
    image_means = np.mean(images, axis=(1, 2))
    kernal_means = np.mean(kernals, axis=(1, 2))

    # reshape means
    image_means = image_means.reshape((images.shape[0], 1, 1))
    image_means = np.tile(image_means, (1, images.shape[1], images.shape[2]))

    kernal_means = kernal_means.reshape((kernals.shape[0], 1, 1))
    kernal_means = np.tile(kernal_means, (1, kernals.shape[1], kernals.shape[2]))

    # shift ('normalize' for lighting changes)
    images = images - image_means
    kernals = kernals - kernal_means

    # convolve
    out = fftconvolve(images, kernals.conj(), mode=mode, axes=[1, 2])

    ############## consider normalizing by standard deviation
    # out = out / (o_img * o_kernal)

    # output
    return out.real

def gaussian(s:np.ndarray) -> float:
    '''
    Compute batched three point guassian

    Args:
        s (np.ndarray) : values (correlation in context)

    Returns:
        dr (float) : required pixel shift to approximate peak
    '''
    # clean values
    s[s <= 0] = 0.0001 # log domain: non-zero postive values

    # compute numerator and denominator (note: index 1 is the center value "i")
    numer = np.log(s[:, 0]) - np.log(s[:, 2])
    denom = 2 * (np.log(s[:, 0]) + np.log(s[:, 2]) - 2 * np.log(s[:, 1]))

    # guassian
    dr = numer / denom

    # output
    return dr

def peak(corr:np.ndarray, precision:type=np.float32) -> tuple[np.ndarray, np.ndarray]:
    '''
    Find batched peak location at the sub pixel level

    Args:
        corr (np.ndarray) : batch of correlation values
        precision (type) : float type to use

    Returns:
        x (np.ndarray) : peak x (column) locations
        y (np.ndarray) : peak y (row) locations
    '''
    # pull key shape values
    n, h, w = corr.shape

    # get max values for each image
    maxs = np.amax(corr, axis=(1, 2))

    # reshape to use in np.where
    maxs = maxs.reshape((n, 1, 1))
    maxs = np.tile(maxs, (1, h, w))

    # get indecies
    inds = np.where(corr == maxs)

    # unpack values
    i, y, x = inds

    # avoid out of bounds for three-point (Note: some accuracy is lost (edge pixels are excluded))
    x[x == 0] = 1
    y[y == 0] = 1

    x[x == w - 1] = w - 2
    y[y == h - 1] = h - 2

    # x indecies for gaussian
    xi = np.tile(i, (3, 1))
    xx = np.vstack([x-1, x, x+1])
    xy = np.tile(y, (3, 1))

    # x correlation values
    xs = corr[xi, xy, xx].astype(precision)
    xs = np.swapaxes(xs, 0, 1)
    
    # x three-point gaussian
    dx = gaussian(xs)

    # y indecies for gaussian
    yi = np.tile(i, (3, 1))
    yx = np.tile(x, (3, 1))
    yy = np.vstack([y-1, y, y+1])

    # y correlation values
    ys = corr[yi, yy, yx].astype(precision)
    ys = np.swapaxes(ys, 0, 1)
    
    # y three-point gaussian
    dy = gaussian(ys)

    # convert types
    x = x.astype(precision)
    y = y.astype(precision)

    # compute new values
    x += dx
    y += dy

    # output
    return x, y

def displacement(corr:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Find the displacement at the sub pixel level

    Args:
        corr (np.ndarray) : batch of correlation values

    Returns:
        u (np.ndarray) : u (column) displacements
        v (np.ndarray) : v (row) displacements
    '''
    # pull key shape values
    n, h, w = corr.shape

    # get peak locations
    x, y = peak(corr)

    # compute displacements
    u = x - w * 0.5 + 0.5
    v = y - h * 0.5 + 0.5

    # output
    return u, v