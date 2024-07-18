import os
import cv2
import numpy as np
import tqdm as tqdm

TOTAL = 'tot'
X = 'x'
Y = 'y'

IMAGE_EXTS = ['.tif', '.jpg', '.png']
VIDEO_EXTS = ['.avi']

class BOS(object):
    '''
    Schlieren class to manage reading, writing, and computing
    '''
    def __init__(self) -> None:
        self._raw = None
        self._computed = None
        self._drawn = None

    def read(self, path:str, append:bool=False) -> None:
        '''
        Read image, directory of images, or video into memory

        Args:
            path (str) : path to file (default=None)
            append (bool) : add image to current data. only works for images (defult=False)

        Returns:
            None
        '''
        # normalize path
        path = os.path.normpath(path)

        # get full path
        path = os.path.abspath(path)

        # data placeholder
        data = []

        # path is a directory
        if os.path.isdir(path):
            # get all file
            files = os.listdir(path)

            # check each one
            for file in files:
                filepath = os.path.join(path, file)

                # only care about images
                if os.path.splitext(filepath)[1] in IMAGE_EXTS:
                    # read image
                    img = cv2.imread(filepath)

                    # check image sizing
                    if len(data) > 0:
                        # match shape
                        if data[0].shape != img.shape:
                            # no match
                            raise ValueError(f'Expected all image to have the same shapes but got shapes {data[0].shape} and {img.shape}')

                    # add image
                    data.append(img)

        # path is single file
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1]

            # if file is an image
            if ext in IMAGE_EXTS:
                # read image
                img = cv2.imread(filepath)

                # add image
                data.append(img)

            # if file is video
            elif ext in VIDEO_EXTS:
                # open video feed
                cap = cv2.VideoCapture(path)

                # read frames
                while cap.isOpened():
                    # get frame
                    ret, frame = cap.read()

                    # stop and end / bad frame
                    if not ret:
                        print(f'Cannot recieve frame {len(data)+1}. Ending')
                        break

                    # save frame
                    data.append(frame)

                # close capture
                cap.release()

                # feedback
                print(f'Read {len(data)} frames')
            
            # not a readable file type
            else:
                raise ValueError(f'Cannot read {ext} files. File must be of the following image types: {IMAGE_EXTS} or video types: {VIDEO_EXTS}')
            
        # not a valid path
        else:
            raise ValueError(f'path does not exist: {path}')

        # append data
        if (append) and (type(self.raw) == np.ndarray):
            # match shape
            if self.raw[0].shape != data[0].shape:
                # no match
                raise ValueError(f'Expected all image to have the same shapes but got shapes {self.raw[0].shape} and {data[0].shape}')

            # append
            self.raw = np.hstack([self.raw, np.array(data)])

        # write data
        else:
            self.raw = np.array(data)

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