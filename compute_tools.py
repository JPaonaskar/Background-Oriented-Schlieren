import numpy as np
from scipy.signal import fftconvolve # potentially faster method with numpy?: https://medium.com/latinxinai/vectorized-convolution-operation-using-numpy-b122fd52fba3

def batch_correlate(images:np.ndarray, kernals:np.ndarray, mode:str='full') -> np.ndarray:
    '''
    Correlation for batches

    Args:
        kernals:np.ndarray : batch of kernal images
        images:np.ndarray : batch of images

    Returns:
        corr (np.ndarray) : batch of corrilation values
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

    # shift
    images = images - image_means
    kernals = kernals - kernal_means

    # convolve
    out = fftconvolve(images, kernals.conj(), mode=mode, axes=[1, 2])

    # output
    return out

def sub_pixel_peak(batch:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Find the location of the peak at the sub pixel level

    Args:
        batch (np.ndarray) : batch of images

    Returns:
        rows (np.ndarray) : peak row locations
        cols (np.ndarray) : peak column locations
    '''
    # get max values for each image
    maxs = np.amax(batch, axis=(1, 2))

    # reshape to use in np.where
    maxs = maxs.reshape((batch.shape[0], 1, 1))
    maxs = np.tile(maxs, (1, batch.shape[1], batch.shape[2]))

    # get indecies
    inds = np.where(batch == maxs)

    # unpack values
    x = inds[1]
    y = inds[2]

    # output
    return x, y