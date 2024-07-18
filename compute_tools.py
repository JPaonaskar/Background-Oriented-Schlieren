import numpy as np
from scipy.signal import fftconvolve # potentially faster method with numpy?: https://medium.com/latinxinai/vectorized-convolution-operation-using-numpy-b122fd52fba3

def batch_correlate(kernals:np.ndarray, images:np.ndarray, mode:str='full') -> np.ndarray:
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

    # normalize
    images = images.astype(np.float32) # - np.mean(images, axis=[1, 2])
    kernals = kernals.astype(np.float32) # - np.mean(kernals, axis=[1, 2])

    # convolve
    out = fftconvolve(images, kernals.conj(), mode=mode, axes=[1, 2])

    # output
    return out

def sub_pixel_peak(batch:np.ndarray) -> tuple[float, float]:
    '''
    Find the location of the peak at the sub pixel level

    Args:
        batch (np.ndarray) : batch of images

    Returns:
        rows (float) : peak row locations
        cols (float) : peak column locations
    '''