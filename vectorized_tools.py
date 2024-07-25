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

# convolution modes
CONV_MODE_FULL = 'full'
CONV_MODE_VALID = 'valid'

def conv2D(images:np.ndarray, kernals:np.ndarray, mode:str=CONV_MODE_FULL) -> np.ndarray:
    '''
    2d Convolution

    Args:
        images (np.ndarray) : batch of images
        kernals (np.ndarray) : batch of kernals
        mode (str) : convolution mode (default=CONV_MODE_FULL)

    Returns:
        out (np.ndarray) : batch of convolutions
    '''
    # 2d convolution using scipy
    out = fftconvolve(images, kernals, mode=mode, axes=[1, 2])

    # output
    return out

def grayscale(images:np.ndarray) -> np.ndarray:
    '''
    Convert a batch of BGR images to grayscale

    Args:
        images (np.ndarray) : batch of images

    Returns:
        out (np.ndarray) : batch of grayscale images
    '''
    # add red channel
    out = images[:, :, :, 2].astype(np.float16) * 0.299

    # add green channel
    out += images[:, :, :, 1].astype(np.float16) * 0.587

    # add blue channel
    out += images[:, :, :, 0].astype(np.float16) * 0.114

    # convert to uint8
    out = out.astype(np.uint8)

    # return
    return out

def batch_subtract(images:np.ndarray, values:np.ndarray) -> np.ndarray:
    '''
    Add scalars accross batch

    Args:
        images (np.ndarray) : batch of images
        values (np.ndarray) : batch of scalar values

    Returns:
        out (np.ndarray) : modified batch
    '''
    # reshape values
    values = values.reshape((images.shape[0], 1, 1))
    values = np.tile(values, (1, images.shape[1], images.shape[2]))

    # add arrays
    images = images - values

    # output
    return images

def batch_multiply(images:np.ndarray, values:np.ndarray) -> np.ndarray:
    '''
    Multiply scalars accross batch

    Args:
        images (np.ndarray) : batch of images
        values (np.ndarray) : batch of scalar values

    Returns:
        out (np.ndarray) : modified batch
    '''
    # reshape values
    values = values.reshape((images.shape[0], 1, 1))
    values = np.tile(values, (1, images.shape[1], images.shape[2]))

    # add arrays
    images = images * values

    # output
    return images

def normxcorr2(images:np.ndarray, kernals:np.ndarray, mode:str=CONV_MODE_FULL) -> np.ndarray:
    '''
    Normalized cross correlation for batches
    Modified version of Sabrewarrior/normxcorr2-python (original code is in /normxcorr2-python)

    Args:
        images (np.ndarray) : batch of images
        kernals (np.ndarray) : batch of kernals
        mode (str) : convolution mode (default=CONV_MODE_FULL)

    Returns:
        corr (np.ndarray) : batch of correlation values
    '''
    # get means
    kernal_means = np.mean(kernals, axis=(1, 2))
    image_means = np.mean(images, axis=(1, 2))

    # subtract means
    kernals = batch_subtract(kernals, kernal_means)
    images = batch_subtract(images, image_means)

    # flip kernals for convolution
    arr = np.flip(np.flip(kernals, axis=1), axis=2)

    # convolution
    out = conv2D(images, arr.conj(), mode=mode)

    # normalization setup
    kernal_ones = np.ones_like(kernals)
    prod = np.prod(kernals.shape) / kernals.shape[0]
    images = conv2D(np.square(images), kernal_ones, mode=mode) - np.square(conv2D(images, kernal_ones, mode=mode)) / prod

    # positive values only
    images[images < 0] = 0

    # normalixe
    kernals = np.sum(np.square(kernals), axis=(1, 2))
    with np.errstate(divide='ignore', invalid='ignore'): 
        out = out / np.sqrt(batch_multiply(images, kernals))

    # remove divisions by zero
    out[~np.isfinite(out)] = 0
    
    # output
    return out

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

def displacement(corr:np.ndarray, precision:type=np.float32) -> tuple[np.ndarray, np.ndarray]:
    '''
    Find displacements at the sub pixel level

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

    # get unique
    i, unique, counts = np.unique(i, return_index=True, return_counts=True) # ISSUE <- returns the furthest upper right value

    # remove non-unique values
    mask = counts == 1

    i = i[mask]
    x = x[unique][mask]
    y = y[unique][mask]

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

    # compute displacements
    u = x - w * 0.5 + 0.5
    v = y - h * 0.5 + 0.5

    # populate data
    temp = np.zeros_like(unique, dtype=float)

    temp[mask] = u
    u = temp.copy()

    temp[mask] = v
    v = temp.copy()

    # output
    return u, v