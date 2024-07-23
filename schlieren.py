'''
SCHLIEREN
by Josha Paonaskar

Background Oriented Schlieren constants, class, methods

Resources:

'''

import os
import cv2
import numpy as np
from tqdm import tqdm

import batch_tools

# display methods
DISP_MAG = 'mag'
DISP_X = 'x'
DISP_Y = 'y'

# datasets
DATA_RAW = 'raw'
DATA_COMPUTED = 'computed'
DATA_DRAWN = 'drawn'

# file extentions
EXTS_IMAGE = ['.tif', '.jpg', '.png']
EXTS_VIDEO = ['.avi']

# keys
KEY_BACKSPACE = 8
KEY_TAB = 9
KEY_ENTER = 13
KEY_ESCAPE = 27
KEY_SPACE = 32

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
            append (bool) : add image to current data. only works for images (default=False)

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
                if os.path.splitext(filepath)[1] in EXTS_IMAGE:
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
            if ext in EXTS_IMAGE:
                # read image
                img = cv2.imread(filepath)

                # add image
                data.append(img)

            # if file is video
            elif ext in EXTS_VIDEO:
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
                raise ValueError(f'Cannot read {ext} files. File must be of the following image types: {EXTS_IMAGE} or video types: {EXTS_VIDEO}')
            
        # not a valid path
        else:
            raise ValueError(f'path does not exist: {path}')

        # append data
        if (append) and (type(self._raw) == np.ndarray):
            # match shape
            if self._raw[0].shape != data[0].shape:
                # no match
                raise ValueError(f'Expected all image to have the same shapes but got shapes {self._raw[0].shape} and {data[0].shape}')

            # append
            self._raw = np.hstack([self._raw, np.array(data)])

        # write data
        else:
            self._raw = np.array(data)

    def compute(self, win_size:int=32, search_size:int=64, space:int=None, start:int=0, stop:int=None, step:int=1, pad:bool=False) -> None:
        '''
        Compute schlieren data

        Args:
            win_size (int) : search windows size (default=32)
            search_size (int) : search size (default=64)
            space (int) : space between referance frame. None implies use start frame (default=None)
            start (int) : starting frame (default=0)
            stop (int) : ending frame (exclusive) (default=None)
            step (int) : step between frames (default=1)
            pad (bool) : pad edges (default=False)

        Returns:
            None
        '''
        # setup slice
        if not stop:
            stop = len(self._raw)
        if space and stop - space > start:
            stop = stop - space

        # slice
        raw_data = self._raw[start:stop:step]

        # unpack shape values
        n, h, w, d = self._raw.shape

        # slice kernals
        if space:
            kernals = raw_data
            raw_data = self._raw[start+space:stop+space:step]

        # tile kernal
        else:
            kernals = np.expand_dims(raw_data[0], axis=0)
            kernals = np.tile(kernals, (n, 1, 1, 1))

        # convert BGR to greyscale if needed
        if d == 3:
            kernals = batch_tools.grayscale(kernals)
            raw_data = batch_tools.grayscale(raw_data)

        # time offset data
        raw_data = raw_data[1:]
        kernals = kernals[:len(kernals)-1]

        # pad raw data
        p = (search_size - win_size) >> 1

        empty = np.zeros((n - 1, h + 2 * p, w + 2 * p), dtype=np.uint8)
        empty[:, p:h+p, p:w+p] = raw_data

        raw_data = empty
        del empty

        # divide into windows
        if pad:
            win_x = np.arange(w // win_size)
            win_y = np.arange(h // win_size)
        else:
            win_x = np.arange((w - 2 * p) // win_size)
            win_y = np.arange((h - 2 * p) // win_size)

        # create list of coordinate pairs
        win_coords = np.meshgrid(win_x, win_y)
        win_coords = np.vstack([win_coords[0].flatten(), win_coords[1].flatten()])
        win_coords = np.swapaxes(win_coords, 0, 1)

        # pre-allocate data (depth is u, v, length)
        data = np.zeros((n, len(win_y), len(win_x), 3))

        # itterate though sections
        for coord in tqdm(win_coords):
            # unpack output location
            row = coord[1]
            col = coord[0]

            # get window location
            win_row = row * win_size
            win_col = col * win_size

            # if no padding
            if not pad:
                win_row += p
                win_col += p

            # pull window
            win = kernals[:, win_row:win_row + win_size, win_col:win_col + win_size]

            # pull search area
            search = raw_data[:, win_row:win_row+win_size + 2 * p, win_col:win_col + win_size + 2 * p]

            # compute correlation and calcualte displacements
            corr = batch_tools.normxcorr2(search, win, mode='full')
            u, v = batch_tools.displacement(corr)

            # store calcualted values
            data[:, row, col, 0] = u
            data[:, row, col, 1] = v
            data[:, row, col, 2] = np.sqrt(np.square(u) + np.square(v))

        # create new computed data if needed
        if (type(self._computed) != np.ndarray) or (self._computed.shape != data.shape):
            self._computed = data

        # preserve old data
        else:
            self._computed[start:stop:step] = data

    def draw(self, method:str=DISP_MAG, thresh:float=5.0, alpha:float=0.6, colormap=cv2.COLORMAP_JET, interplolation=cv2.INTER_NEAREST, start:int=0, stop:int=None, step:int=1) -> None:
        '''
        Draw computed data

        Args:
            method (str) : drawing method (default=DISP_MAG)
            thresh (float) : value maximum (defult=5.0)
            alpha (float) : blending between raw and computed (defult=0.6)
            colormap (int) : colormap (default=cv2.COLORMAP_JET)
            interplolation (int) : interplolation method (default=cv2.INTER_NEAREST)
            start (int) : starting frame (default=0)
            stop (int) : ending frame (exclusive) (default=None)
            step (int) : step between frames (default=1)

        Returns:
            None
        '''
        # setup slice
        if not stop:
            stop = len(self._raw)

        # slice data
        drawn = self._raw[start:stop:step]
        data = self._computed[start:stop:step]

        # store shape
        n, h, w, d = drawn.shape
        
        # convert raw data to BGR in needed
        if d == 1:
            drawn = np.stack([drawn, drawn, drawn], axis=3)

        # get computed data
        if method == DISP_X:
            data = data[:, :, :, 0]
        elif method == DISP_Y:
            data = data[:, :, :, 1]
        elif method == DISP_MAG:
            data = data[:, :, :, 2]
        else:
            raise ValueError(f'Method {method} is not a valid method')
        
        # apply threshold
        mask = data > thresh
        data[mask] = np.nan

        # normalize data
        data = (data * 255 / thresh).astype(np.uint8)

        # draw images
        for i in tqdm(range(n)):
            point = data[i]
            raw = drawn[i]

            # apply colormap
            point = cv2.applyColorMap(point, cv2.COLORMAP_JET).astype(np.float16)

            # mask colormap
            point[mask[i]] = np.nan

            # resize
            point = cv2.resize(point, (w, h), interpolation=interplolation)

            # replace empty values
            nans = np.isnan(point)
            point[nans] = raw[nans]

            # belnd and store
            drawn[i] = (raw * alpha + point * (1 - alpha)).astype(np.uint8)

        # create new drawn data if needed
        if (type(self._drawn) != np.ndarray) or (self._drawn.shape != drawn.shape):
            self._drawn = drawn

        # preserve old data
        else:
            self._drawn[start:stop:step] = drawn

    def display(self, dataname:str=DATA_DRAWN) -> None:
        '''
        Display drawn data

        Args:
            None

        Returns:
            None
        '''
        # get images
        imgs = None
        if dataname == DATA_RAW:
            imgs = self._raw
        elif dataname == DATA_COMPUTED:
            imgs = self._computed
        elif dataname == DATA_DRAWN:
            imgs = self._drawn
        else:
            ValueError(f'{dataname} in not a valid dataset')

        # shift unsigned 8-bit
        imgs = imgs - np.min(imgs) # 0.0 - max
        imgs = imgs * 255.0 / np.max(imgs) # 0.0 - 255.0
        imgs = imgs.astype(np.uint8) # 8bit

        # placeholders
        ind = 0

        # set window
        cv2.namedWindow(dataname)

        # loop
        while True:
            # load new image
            img = imgs[ind]

            # resize
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

            # draw index
            #cv2

            # draw frame
            cv2.imshow(dataname, img)

            # keys
            k = cv2.waitKey(0)

            # quit
            if (k == ord('q')) or (k == KEY_ESCAPE):
                break

            # looping
            elif k == ord('a'):
                if ind > 0:
                    ind -= 1
            elif k == ord('d'):
                if ind < len(imgs) - 1:
                    ind += 1

            else:
                print('pressed:', k)

        # close window
        cv2.destroyWindow(dataname)

    def write(self, path:str=None, start:int=0, stop:int=None, step:int=1) -> None:
        '''
        Write image or video

        Args:
            path (str) : path to save location (direcory or file) (default=None)
            start (int) : starting frame (default=0)
            stop (int) : ending frame (exclusive) (default=None)
            step (int) : step between frames (default=1)

        Returns:
            None
        '''