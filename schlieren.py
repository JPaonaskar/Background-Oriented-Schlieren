import numpy as np
import tqdm as tqdm

TOTAL = 'tot'
X = 'x'
Y = 'y'

class BOS(object):
    '''
    Schlieren class to manage reading, writing, and computing
    '''
    def __init__(self) -> None:
        self._raw = None
        self._computed = None
        self._drawn = None

    def read(self, filename:str, append:bool=False) -> None:
        '''
        Read image, directory of images, or video into memory

        Args:
            filename (str) : path to file (default=None)
            append (bool) : add image to current data. only works for images (defult=False)

        Returns:
            None
        '''

    def write(self, path:str=None, start:int=0, stop:int=None, step:int=1) -> None:
        '''
        Write image or video

        Args:
            path (str) : path to save location (direcory or file) (default=None)
            start (int) : starting frame (defult=0)
            stop (int) : ending frame (exclusive) (defult=None)
            step (int) : step between frames (defult=1)

        Returns:
            None
        '''

    def compute(self, win_size:int=32, search_size:int=64, start:int=0, stop:int=None, step:int=1) -> None:
        '''
        Compute schlieren data

        Args:
            win_size (int) : search windows size (default=32)
            search_size (int) : search size (default=64)
            start (int) : starting frame (defult=0)
            stop (int) : ending frame (exclusive) (defult=None)
            step (int) : step between frames (defult=1)

        Returns:
            None
        '''

    def draw(self, method:str=TOTAL, thresh:float=4.0, alpha:float=0.6, start:int=0, stop:int=None, step:int=1) -> None:
        '''
        Draw computed data

        Args:
            method (str) : drawing method (default=32)
            start (int) : starting frame (defult=0)
            stop (int) : ending frame (exclusive) (defult=None)
            step (int) : step between frames (defult=1)

        Returns:
            None
        '''

    def display(self) -> None:
        '''
        Display drawn data

        Args:
            None

        Returns:
            None
        '''