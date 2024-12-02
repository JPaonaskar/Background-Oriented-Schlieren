'''
SCHLIEREN
by Josha Paonaskar

Background Oriented Schlieren constants, class, methods

Resources:
    https://web.mit.edu/dphart/www/Super-Resolution%20PIV.pdf
'''

import os
import cv2
import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

import vectorized_tools

# display methods
DISP_MAG = 'mag'
DISP_X = 'x'
DISP_Y = 'y'

# datasets
DATA_RAW = 'raw'
DATA_SPLIT = 'split'
DATA_COMPUTED = 'computed'
DATA_DRAWN = 'drawn'

# file extensions
EXTS_IMAGE = ['.tif', '.jpg', '.png']
EXTS_VIDEO = ['.avi', '.MOV', '.mp4']
EXTS_AVI = '.avi'
EXTS_NUMPY = '.npy'
EXTS_JPIV = '.jvc'

# keys
KEY_EXIT = -1
KEY_BACKSPACE = 8
KEY_TAB = 9
KEY_ENTER = 13
KEY_ESCAPE = 27
KEY_SPACE = 32

# colors
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

# interpolation
INTER_NEAREST = cv2.INTER_NEAREST
INTER_CUBIC = cv2.INTER_CUBIC

# pairing methods
PAIR_CASCADE = 'cascade'
PAIR_PAIRS = 'pairs'
PAIR_CONSECUTIVE = 'consecutive'

# color maps
COLORMAP_COMPUTED = -1
COLORMAP_AUTUMN = 0
COLORMAP_BONE = 1
COLORMAP_JET = 2
COLORMAP_WINTER = 3
COLORMAP_RAINBOW = 4
COLORMAP_OCEAN = 5
COLORMAP_SUMMER = 6
COLORMAP_SPRING = 7
COLORMAP_COOL = 8
COLORMAP_HSV = 9
COLORMAP_PINK = 10
COLORMAP_HOT = 11
COLORMAP_PARULA = 12
COLORMAP_MAGMA = 13
COLORMAP_INFERNO = 14
COLORMAP_PLASMA = 15
COLORMAP_VIRIDIS = 16
COLORMAP_CIVIDIS = 17
COLORMAP_TWILIGHT = 18
COLORMAP_TWILIGHT_SHIFTED = 19
COLORMAP_TURBO = 20
COLORMAP_DEEPGREEN = 21


def _spiral_coords(w:int, h:int) -> np.ndarray:
    '''
    Create a list of coordinates spiraling outward from the center

    Args:
        w (int) : width
        h (int) : height

    Returns:
        coords (np.ndarray) : list of coordinates
    '''
    # initial state
    x = w // 2
    y = h // 2
    d = 0 # 0 = RIGHT, 1 = DOWN, 2 = LEFT, 3 = UP
    s = 1 # chain size

    # get furthest coord from center
    dist = max(x, y)

    # coordinate output
    coords = [[x, y]]

    # begin spiral (for number of step out from center) (Alternates from WS and WN)
    for i in range(2*dist + 1):
        # loop twice (chains are the same size twice)
        for j in [0, 1]:
            # step across chain
            for k in range(s):
                # move in direction
                if (d == 0):
                    x += 1
                elif (d == 1):
                    y += 1
                elif (d == 2):
                    x -= 1
                elif (d == 3):
                    y -= 1

                # check if in bounds
                if (x >= 0 and x < w and y >= 0 and y < h):
                    # save coordinate
                    coords.append([x, y])

            # change direction
            d = (d + 1) % 4

        # increase chain size
        s += 1

    # return
    return np.array(coords, dtype=int)


class BOS(object):
    '''
    Schlieren class to manage reading, writing, and computing

    Args: 
        None

    Returns:
        object (BOS) : BOS instance
    '''
    def __init__(self) -> None:
        self._raw = None
        self._references = None
        self._images = None
        self._computed = None
        self._drawn = None

    def read(self, path:str, computed:bool=False) -> None:
        '''
        Read image, directory of images, or video into memory

        Args:
            path (str) : path to file (default=None)
            computed (bool) : images are computed results (default=False)

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
            for file in tqdm(files):
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
                img = cv2.imread(path)

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
                        print(f'Cannot receive frame {len(data)+1}. Ending')
                        break

                    # save frame
                    data.append(frame)

                # close capture
                cap.release()

                # feedback
                print(f'Read {len(data)} frames')

            # if path is numpy
            elif ext == EXTS_NUMPY:
                with open(path, 'rb') as f:
                    data = np.load(f)
            
            # not a readable file type
            else:
                raise ValueError(f'Cannot read {ext} files. File must be of the following image types: {EXTS_IMAGE} or video types: {EXTS_VIDEO}')
            
        # not a valid path
        else:
            raise ValueError(f'path does not exist: {path}')

        # write data
        if computed:
            self._computed = np.array(data)
        else:
            self._raw = np.array(data, dtype=np.uint8)

    def read_jpiv(self, path:str, stacked:bool=True) -> None:
        '''
        Read JPIV result file into computed data

        Args:
            path (str) : path to JPIV file, will read stored image if named the same
            stacked (bool) : image is two frames stacked (default=True)

        Returns:
            None
        '''
        # normalize path
        path = os.path.normpath(path)

        # get full path
        path = os.path.abspath(path)
        
        # validate path
        if not os.path.isfile(path):
            raise FileNotFoundError(f'File does not exist: {path}')
        elif os.path.splitext(path)[1] != EXTS_JPIV:
            raise ValueError(f'Expected {EXTS_JPIV} file but got {os.path.basename(path)}')
        
        # read file
        data = []
        with open(path, 'r') as f:
            # read lines
            line = f.readline()
            while line:
                line_data = line.strip().split(' ')

                # convert to numerical values
                line_values = []
                for value in line_data:
                    line_values.append(float(value))

                # save to data
                data.append(line_values)

                # read new line
                line = f.readline()

        # convert to numpy
        data = np.array(data)

        # slice data
        x = data[:, 0].astype(np.int16)
        y = data[:, 1].astype(np.int16)
        u = data[:, 2]
        v = data[:, 3]

        # remove bad values
        mask = np.bitwise_and(u == 1, v == 0)
        u[mask] = np.nan
        v[mask] = np.nan

        # calulate magnitude
        m = np.sqrt(np.square(u) + np.square(v))

        # get step sizes
        dx = np.unique(x)
        dx.sort()
        dx = np.amin(dx[1:] - dx[:-1])

        dy = np.unique(y)
        dy.sort()
        dy = np.amin(dy[1:] - dy[:-1])

        # transform coordinates
        x = x // dx
        y = y // dy

        # get padding
        padx = x.min()
        pady = y.min()

        # create image (ovewrite data)
        w = (x.max() + padx)
        h = (y.max() + pady)
        data = np.zeros((1, h, w, 3))

        # store data points
        data[0, y, x, 0] = u
        data[0, y, x, 1] = v
        data[0, y, x, 2] = m

        # save to computed data
        self._computed = data

        # get other files
        directory = os.path.dirname(path)
        files = os.listdir(directory)

        # check for match
        name = os.path.splitext(os.path.basename(path))[0]
        image_path = None

        for file in files:
            # image name matches file name
            filename, ext = os.path.splitext(file)
            if (filename in name) and (ext in EXTS_IMAGE):
                # save and stop looking
                image_path = os.path.join(directory, file)
                break

        # if image file was found
        if image_path:
            # feedback
            print(f'Found image {os.path.basename(image_path)}')

            # read image file
            image = cv2.imread(image_path)

            # split stacked
            if stacked:
                # unpack shape
                h, w, _ = image.shape

                # slice
                images = np.zeros((2, h // 2, w, 3), dtype=np.uint8)

                # store each image
                images[0] = image[0:h//2, :, :]
                images[1] = image[h//2:, :, :]

            # other methods
            else:
                NotImplementedError('Currently does not support non-stacked images')
        else:
            # feedback
            print('No image found')

            # open blank
            images = np.zeros((2, h, w, 3))

        # save image
        self._raw = images

    def gaussianBlur(self, ksize:tuple=(3,3), sigmaX:float=0, sigmaY:float=0) -> None:
        '''
        Apply gaussian blur to raw images

        Args:
            ksize (tuple) : kernel shape (default=(3, 3))
            sigmaX (float) : gaussian kernel standard deviation in the X (default=0)
            sigmaY (float) : gaussian kernel standard deviation in the Y (default=0)

        Returns:
            None
        '''
        # blur images
        print('Gaussian Blur')
        for i in tqdm(range(len(self._raw))):
            self._raw[i] = cv2.GaussianBlur(self._raw[i], ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)

    def medianBlur(self, ksize:int=3) -> None:
        '''
        Apply gaussian blur to raw images

        Args:
            ksize (tuple) : kernel size (default=3)

        Returns:
            None
        '''
        # blur images
        print('Median Blur')
        for i in tqdm(range(len(self._raw))):
            self._raw[i] = cv2.medianBlur(self._raw[i], ksize=ksize)

    def split(self, start:int=0, stop:int=None, step:int=1, method:str=PAIR_CASCADE) -> None:
        '''
        Split data into image pairs

        Args:
            start (int) : starting frame (default=0)
            stop (int) : stopping frame (default=None)
            step (int) : step between frames (default=1)
            method (str) : pairing method (default=PAIR_CASCADE)

        Returns:
            None
        '''
        # setup slice
        if not stop:
            stop = len(self._raw) + 1

        # slice images
        images = self._raw[start:stop:step]

        # make cascade pairs
        if method == PAIR_CASCADE:
            # store images
            self._images = images[1:]

            # use first image a reference
            self._references = images[0]
            self._references = np.tile(self._references, (len(self._images), 1, 1, 1))

        # make consecutive pairs
        elif method == PAIR_CONSECUTIVE:
            # use prior image a first
            self._references = images[0:-1]

            # use next image a image
            self._images = images[1:]

        # make pairs
        elif method == PAIR_PAIRS:
            # use first image a reference
            self._references = images[0::2]

            # use second image as image
            self._images = images[1::2]

        else:
            raise ValueError(f'{method} is not a valid pairing method')

    def _setup_compute(self, win_size:int, search_size:int, overlap:int, pad:bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        '''
        Setup data and other values for computing

        Args:
            win_size (int) : search windows size
            search_size (int) : search size
            overlap (int) : overlap between windows
            pad (bool) : pad edges

        Returns:
            img_data (np.ndarray) : compute images
            ref_data (np.ndarray) : reference images
            win_x (np.ndarray) : x axis
            win_y (np.ndarray) : y axis
            n (int) : number of data points
            p (int) : padding size
        '''
        print('Setting Up Compute')
        # unpack
        ref_data = self._references.copy()
        img_data = self._images.copy()

        # unpack shape values
        n, h, w, d = ref_data.shape

        print('Converting to Grayscale')

        # convert BGR to grayscale if needed
        if d == 3:
            ref_data = vectorized_tools.grayscale(ref_data)
            img_data = vectorized_tools.grayscale(img_data)

        print('Padding')

        # pad raw data
        p = (search_size - win_size) >> 1

        empty = np.zeros((n, h + 2 * p, w + 2 * p), dtype=np.uint8)
        empty[:, p:h+p, p:w+p] = img_data

        img_data = empty
        del empty

        print('Dividing into Windows')

        # add padding
        if not pad:
            w -= 2 * p
            h -= 2 * p

        # maximum window location (inclusive)
        x1 = (w - win_size) // (win_size - overlap)
        y1 = (h - win_size) // (win_size - overlap)

        # error values
        ex = (w - win_size) % (win_size - overlap) // 2
        ey = (h - win_size) % (win_size - overlap) // 2

        # build window coordinates
        win_x = np.arange(x1 + 1)
        win_y = np.arange(y1 + 1)

        # convert to pixel coordinates
        win_x = win_x * (win_size - overlap) + ex
        win_y = win_y * (win_size - overlap) + ey

        # add padding
        if not pad:
            win_x += p
            win_y += p

        # return key values
        return img_data, ref_data, win_x, win_y, n, p

    def compute(self, win_size:int=32, search_size:int=64, overlap:int=0, pad:bool=False) -> None:
        '''
        Compute schlieren data

        Args:
            win_size (int) : search windows size (default=32)
            search_size (int) : search size (default=64)
            overlap (int) : overlap between windows (default=0)
            pad (bool) : pad edges (default=False)

        Returns:
            None
        '''
        # get compute values
        raw_data, kernels, win_x, win_y, n, p = self._setup_compute(win_size, search_size, overlap, pad)

        # create list of coordinate pairs
        win_coords = np.meshgrid(np.arange(len(win_x)), np.arange(len(win_y)))
        win_coords = np.vstack([win_coords[0].flatten(), win_coords[1].flatten()])
        win_coords = np.swapaxes(win_coords, 0, 1)

        # pre-allocate data (depth is u, v, length)
        data = np.zeros((n, len(win_y), len(win_x), 3))

        # task for threading
        def process(coord:np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
            # unpack output location
            row = coord[1]
            col = coord[0]

            # get window location
            win_row = win_y[row]
            win_col = win_x[col]

            # pull window
            win = kernels[:, win_row:win_row + win_size, win_col:win_col + win_size]

            # pull search area
            search = raw_data[:, win_row:win_row+win_size + 2 * p, win_col:win_col + win_size + 2 * p]

            # compute correlation and calculate displacements
            corr = vectorized_tools.normxcorr2(search, win, mode='full')
            u, v = vectorized_tools.displacement(corr)

            # compute magnitude
            m = np.sqrt(np.square(u) + np.square(v))

            # output needed information to save data
            return row, col, u, v, m
        
        # threading for speed
        with ThreadPoolExecutor() as executor:
            futures = []

            # iterate though sections
            print('Starting Processes')
            for coord in tqdm(win_coords):
                # start process
                futures.append(executor.submit(process, coord))

            # get frames
            print("Computing Frames")
            for future in tqdm(futures):
                # add frame
                row, col, u, v, m = future.result()

                # store calcualted values
                data[:, row, col, 0] = u
                data[:, row, col, 1] = v
                data[:, row, col, 2] = m

        # store computed data
        self._computed = data

    def compute_multi(self, win_size:int=32, search_size:int=64, particle_size:int=2, overlap:int=0, pad:bool=False) -> None:
        '''
        Mulipass compute

        Args:
            win_size (int) : search windows size (default=32)
            search_size (int) : search size (default=64)
            particle_size (int) : particle size (default=2)
            overlap (int) : overlap between windows (default=0)
            pad (bool) : pad edges (default=False)

        Returns:
            None
        '''
        raise NotImplementedError('This function is not implemented yet')

    def draw(self, method:str=DISP_MAG, thresh:float=5.0, alpha:float=0.6, colormap:int=COLORMAP_JET, interpolation:int=INTER_NEAREST, masked:float=None) -> None:
        '''
        Draw computed data

        Args:
            method (str) : drawing method (default=DISP_MAG)
            thresh (float) : value maximum (default=5.0)
            alpha (float) : blending between raw and computed (default=0.6)
            colormap (int) : colormap (default=COLORMAP_JET)
            interpolation (int) : interpolation method (default=INTER_NEAREST)
            masked (float) : treat low displacements as a mask (default=None)

        Returns:
            None
        '''
        # unpack data
        data = self._computed.copy().astype(np.float16)

        if type(self._images) == np.ndarray:
            drawn = self._images.copy()
        else:
            drawn = np.zeros_like(data, dtype=np.uint8)
            
        # time offset images
        if drawn.shape[0] > data.shape[0]:
            drawn = drawn[1:]

        # store shape
        n, h, w, d = drawn.shape
        
        # convert raw data to BGR in needed
        if d == 1:
            drawn = np.stack([drawn, drawn, drawn], axis=3)

        # get computed data
        if colormap != COLORMAP_COMPUTED:
            if method == DISP_X:
                data = data[:, :, :, 0]
            elif method == DISP_Y:
                data = data[:, :, :, 1]
            elif method == DISP_MAG:
                data = data[:, :, :, 2]
            else:
                raise ValueError(f'Method {method} is not a valid method')
        
        # apply threshold
        mask = np.abs(data) > thresh

        # remove zeros
        if masked:
            mask = np.bitwise_or(mask, np.abs(data) <= masked)

        # mask
        data[mask] = 0.0

        # collapse mask
        if len(mask.shape) == 4 and mask.shape[3] > 1:
            copy = mask.copy().astype(np.int8)

            # join layers
            mask = copy[:, :, :, 0]
            for i in range(1, copy.shape[3]):
                mask = cv2.bitwise_or(mask, copy[:, :, :, 1])

            # delete copy
            del copy

        # max for magnitude values
        if (method == DISP_MAG) and (colormap != COLORMAP_COMPUTED):
            data = (np.abs(data) * 255 / thresh).astype(np.uint8)
        
        # min max for negative values
        else:
            data = (data * 127.5 / thresh + 127.5).astype(np.uint8)

        # draw images
        print('Drawing Frames')
        for i in tqdm(range(n)):
            point = data[i]
            raw = drawn[i]

            # apply colormap
            if colormap != COLORMAP_COMPUTED:
                point = cv2.applyColorMap(point, colormap)

            # make blending image
            blend = mask[i].astype(np.uint8) * 255
            blend = np.stack([blend, blend, blend], axis=2)

            # resize
            blend = cv2.resize(blend, (w, h), interpolation=interpolation)
            point = cv2.resize(point, (w, h), interpolation=interpolation)

            # modify to be alpha
            blend = blend / 255.0 * alpha + (1 - alpha)

            # blend and store
            drawn[i] = (raw * blend + point * (1 - blend)).astype(np.uint8)

        # store drawn data
        self._drawn = drawn

    def quiver(self, thresh:float=5, alpha:float=0.3, colormap:int=COLORMAP_JET, thickness:int=10, interpolation:int=INTER_NEAREST, scale:float=5.0):
        '''
        Quiver plot

        Args:
            thresh (float) : value maximum (default=5.0)
            alpha (float) : blending between image and arrows (default=0.3)
            colormap (int) : colormap (default=COLORMAP_JET)
            thickness (int) : arrow thickness (default=10)
            interpolation (int) : interpolation method (default=INTER_NEAREST)
            scale (float) : image scaling to make arrows more visible (default=5.0)

        Returns:
            None
        '''
        # unpack data
        data = self._computed.copy().astype(np.float16)

        if type(self._images) == np.ndarray:
            images = self._images.copy()
        else:
            images = np.zeros_like(data, dtype=np.uint8)
        
        # time offset images
        if images.shape[0] > data.shape[0]:
            images = images[1:]

        # store shape
        n, h, w, d = images.shape
        h = round(h * scale)
        w = round(w * scale)

        # create drawn array
        drawn = np.zeros((n, h, w, 3), dtype=np.uint8)
        
        # convert raw data to BGR in needed
        if d == 1:
            images = np.stack([images, images, images], axis=3)
        
        # apply threshold
        mask = np.abs(data) > thresh
        data[mask] = 0.0
        
        # min max data
        data = (data * 127.5 / thresh + 127.5).astype(np.uint8)

        # calculate coordinates
        _, _h, _w, _ = data.shape
        dh = h / _h
        dw = w / _w

        # draw images
        print('Quivering Frames')
        for i in tqdm(range(n)):
            img = images[i]

            # get colormap
            colors = data[i]
            if colormap != COLORMAP_COMPUTED:
                colors = cv2.applyColorMap(data[i], colormap)

            # resize
            img = cv2.resize(img, (w, h), interpolation=interpolation)
            arrows = np.zeros_like(img)

            # draw quiver
            for j in range(_h):
                for k in range(_w):
                    # get vector properties
                    Y = round(dh * (0.5 + j))
                    X = round(dw * (0.5 + k))
                    U = round(dw * (data[i, j, k, 0] / 255.0 - 0.5))
                    V = round(dh * (data[i, j, k, 1] / 255.0 - 0.5))

                    C = (int(colors[j, k, 0]), int(colors[j, k, 1]), int(colors[j, k, 2]))

                    # plot vector
                    arrows = cv2.arrowedLine(arrows, (X, Y), (X + U, Y + V), color=C, thickness=thickness)

            # blend and store
            drawn[i] = (img * alpha + arrows * (1 - alpha)).astype(np.uint8)

        # store drawn data
        self._drawn = drawn

    def _get_data(self, dataname:str=DATA_DRAWN, normalize:bool=True) -> tuple[np.ndarray, np.ndarray]:
        '''
        Get humanized data

        Args:
            dataname (str) : data to display (default=DATA_DRAWN)
            normalize (bool) : normalize data to 0-255 (default=True)

        Returns:
            data (np.ndarray) : unpacked data
            indexes (np.ndarray) : pair indexes
        '''
        # get images
        data = None
        indexes = None
        if dataname == DATA_RAW:
            # get raw data
            data = self._raw.copy()

            # create indexes
            indexes = np.arange(len(data), dtype=np.uint16)

        elif dataname == DATA_SPLIT:
            n_ref, h, w, d = self._references.shape
            n_img = self._images.shape[0]

            # create empty array
            data = np.zeros((n_ref + n_img, h, w, d), dtype=np.uint8)

            # assign data
            data[0::2] = self._references.copy()
            data[1::2] = self._images.copy()

            # create indexes
            indexes = np.zeros((n_ref + n_img), dtype=np.uint16)

            # assign indexes
            indexes[0::2] = np.arange(n_ref)
            indexes[1::2] = np.arange(n_img)

        elif dataname == DATA_COMPUTED:
            # get compuited data
            data = self._computed.copy()

            # create indexes
            indexes = np.arange(len(data), dtype=np.uint16)

        elif dataname == DATA_DRAWN:
            # get drawn data
            data = self._drawn.copy()

            # create indexes
            indexes = np.arange(len(data), dtype=np.uint16)
        else:
            ValueError(f'{dataname} in not a valid dataset')

        # normalize
        if normalize:
            # shift unsigned 8-bit
            data = data.astype(np.float32)
            data = data - np.nanmin(data) # 0.0 - max
            data = data * 255.0 / np.nanmax(data) # 0.0 - 255.0
            data = data.astype(np.uint8) # 8bit

            # replace nan with mean
            data = np.nan_to_num(data, nan=np.mean(data))

        # output
        return data, indexes

    def display(self, dataname:str=DATA_DRAWN, font:int=cv2.FONT_HERSHEY_SIMPLEX, font_scale:float=0.5, font_color:tuple[int, int, int]=COLOR_WHITE, font_thickness:int=1, font_pad:int=8, normalize:bool=True, scale:float=-1.0) -> None:
        '''
        Display drawn data

        Args:
            dataname (str) : data to display (default=DATA_DRAWN)
            font (int) : overlay font, None displays no text (default=cv2.FONT_HERSHEY_SIMPLEX)
            font_scale (float) : overlay font scale (default=0.5)
            font_color (tuple[int, int, int]) : overlay font color (default=COLOR_WHITE)
            font_thickness (int) : overlay font thickness (default=1)
            font_pad (int) : overlay font padding from edges (default=8)
            normalize (bool) : normalize image (default=True)
            scale (float) : image resizing, less then zero fixes (512 x 512) (default=-1.0)

        Returns:
            None
        '''
        # get data
        imgs, inds = self._get_data(dataname=dataname, normalize=normalize)

        # placeholders
        ind = 0

        # set window
        cv2.namedWindow(dataname)

        # loop
        while True:
            # load new image
            img = imgs[ind]

            # resize
            if scale < 0:
                img = cv2.resize(img, (512, 512), interpolation=INTER_NEAREST)
            elif scale == 1.0:
                pass
            else:
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=INTER_NEAREST)

            # draw index
            if font != None:
                # get text size
                text = f'{inds[ind]+1} / {inds.max()+1}'
                (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                # put text
                org = (img.shape[1] - w - font_pad, img.shape[0] - h - font_pad)
                img = cv2.putText(img, text, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # draw frame
            cv2.imshow(dataname, img)

            # keys
            k = cv2.waitKey(0)

            # quit
            if (k == ord('q')) or (k == KEY_ESCAPE) or (k == KEY_EXIT):
                break

            # large stepping
            elif k == ord('a'):
                # full step
                if ind >= 10:
                    ind -= 10
                # floor
                else:
                    ind = 0
            elif k == ord('d'):
                # full step
                if ind < len(imgs) - 10:
                    ind += 10
                # ceil
                else:
                    ind = len(imgs) - 1

            # small stepping
            elif k == ord(','):
                if ind > 0:
                    ind -= 1
            elif k == ord('.'):
                if ind < len(imgs) - 1:
                    ind += 1

            else:
                print('pressed:', k, end=' ')
                if k >= 0:
                    print(chr(k).strip())
                else:
                    print()

        # close window
        cv2.destroyWindow(dataname)
    
    def _live_render_cell(self, win:np.ndarray, search:np.ndarray, method:str=DISP_MAG, thresh:float=5.0, alpha:float=0.6, colormap:int=COLORMAP_JET, masked:bool=False) -> np.ndarray:
        '''
        Compute and draw a cell

        Args:
            win (np.ndarray) : window to search for
            search (np.ndarray) : search area
            method (str) : drawing method (default=DISP_MAG)
            thresh (float) : value maximum (default=5.0)
            alpha (float) : blending between raw and computed (default=0.6)
            colormap (int) : colormap (default=COLORMAP_JET)
            masked (float) : treat low displacements as a mask (default=None)

        Returns:
            cell (np.ndarray) : drawn cell
        '''
        # create background
        _, y, x = search.shape
        _, w, h = win.shape

        x = (x - w) // 2
        y = (y - h) // 2

        cell = search[:, y:y+h, x:x+w].copy()
        cell = np.stack([cell, cell, cell], axis=3)

        # compute correlation and calcualte displacements
        corr = vectorized_tools.normxcorr2(search, win, mode='full')
        u, v = vectorized_tools.displacement(corr)

        # calculate magnitude
        m = np.sqrt(np.square(u) + np.square(v))

        # get computed data
        if method == DISP_X:
            data = u
        elif method == DISP_Y:
            data = v
        elif method == DISP_MAG:
            data = m

        # apply threshold
        mask = data > thresh

        # apply mask
        if masked:
            mask = np.bitwise_or(mask, data <= masked)

        # normalize data
        data = (data * 255 / thresh).astype(np.uint8)

        # apply colormap
        data = cv2.applyColorMap(data, colormap).astype(np.float16)[0]

        # blend and store
        cell[~mask] = (cell[~mask] * alpha + data * (1 - alpha)).astype(np.uint8)

        # output
        return cell

    def live(self, win_size:int=32, search_size:int=64, overlap:int=0, pad:bool=False, save_win_size:int=32, save_search_size:int=64, save_overlap:int=0, method:str=DISP_MAG, thresh:float=5.0, alpha:float=0.6, colormap:int=COLORMAP_JET, interpolation=INTER_NEAREST, masked:float=None, font:int=cv2.FONT_HERSHEY_SIMPLEX, font_scale:float=0.5, font_color:tuple[int, int, int]=COLOR_WHITE, font_thickness:int=1, font_pad:int=8) -> None:
        '''
        Live computing and rendering

        Args:
            win_size (int) : search windows size (default=32)
            search_size (int) : search size (default=64)
            overlap (int) : overlap between windows (default=0)
            pad (bool) : pad edges (default=False)

            save_win_size (int) : search windows size for saving (default=32)
            save_search_size (int) : search size for saving (default=64)
            save_overlap (int) : overlap between windows for saving (default=0)

            method (str) : drawing method (default=DISP_MAG)
            thresh (float) : value maximum (default=5.0)
            alpha (float) : blending between raw and computed (default=0.6)
            colormap (int) : colormap (default=COLORMAP_JET)
            interpolation (int) : interpolation method (default=INTER_NEAREST)
            masked (float) : treat low displacements as a mask (default=None)

            font (int) : overlay font, None displays no text (default=cv2.FONT_HERSHEY_SIMPLEX)
            font_scale (float) : overlay font scale (default=0.5)
            font_color (tuple[int, int, int]) : overlay font color (default=COLOR_WHITE)
            font_thickness (int) : overlay font thickness (default=1)
            font_pad (int) : overlay font padding from edges (default=8)

        Returns:
            None
        '''
        # unpack image size
        _, h, w, _ = self._raw.shape

        # setup computing values
        raw_data, kernels, win_x, win_y, n, p = self._setup_compute(win_size, search_size, overlap, None, pad)

        # create spiral
        coords = _spiral_coords(len(win_x), len(win_y))

        # create data lists values
        drawn = np.zeros((n-1, h, w, 3), dtype=np.uint8)

        # placeholders
        ind = 0
        coord_inds = np.zeros((n - 1), dtype=int)

        # set window
        cv2.namedWindow('Live')

        # loop
        while True:
            # do computing
            if coord_inds[ind] < len(coords):
                # unpack output location
                row = coords[coord_inds[ind], 1]
                col = coords[coord_inds[ind], 0]

                # get window location
                win_row = win_y[row]
                win_col = win_x[col]

                # pull window
                win = kernels[ind:ind+1, win_row:win_row + win_size, win_col:win_col + win_size]

                # pull search area
                search = raw_data[ind:ind+1, win_row:win_row+win_size + 2 * p, win_col:win_col + win_size + 2 * p]

                # compute correlation and calcualte displacements
                cell = self._live_render_cell(win, search, method, thresh, alpha, colormap, masked)

                # draw cell
                drawn[ind:ind+1, win_row:win_row + win_size, win_col:win_col + win_size, :] = cell

                # move index
                coord_inds[ind] += 1

            # load image
            img = drawn[ind]

            # resize
            img = cv2.resize(img, (512, 512), interpolation=INTER_NEAREST)

            # draw index
            if font != None:
                # get text size
                text = f'{ind+1} / {len(drawn)}'
                (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                # put text
                org = (img.shape[1] - w - font_pad, img.shape[0] - h - font_pad)
                img = cv2.putText(img, text, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # draw frame
            cv2.imshow('Live', img)

            # keys
            k = cv2.waitKey(1)

            # quit
            if (k == ord('q')) or (k == KEY_ESCAPE):# or (k == KEY_EXIT):
                break

            # large stepping
            elif k == ord('a'):
                # full step
                if ind >= 10:
                    ind -= 10
                # floor
                else:
                    ind = 0
            elif k == ord('d'):
                # full step
                if ind < len(drawn) - 10:
                    ind += 10
                # ceil
                else:
                    ind = len(drawn) - 1

            # small stepping
            elif k == ord(','):
                if ind > 0:
                    ind -= 1
            elif k == ord('.'):
                if ind < len(drawn) - 1:
                    ind += 1

            # save
            elif k == ord('s'):
                # get slices
                raise NotImplementedError("Saving single frames is not implemented yet (need single slicing)")

                # compute and draw frame
                self.compute(win_size=save_win_size, search_size=save_search_size, overlap=save_overlap, pad=pad)
                self.draw(method=method, thresh=thresh, alpha=alpha, colormap=colormap, interpolation=interpolation, masked=masked)
                
                # save frame
                self.write('results')
                print('Frame Saved')

            # save stacked image for jpiv
            elif k == ord('j'):
                # get slices
                raise NotImplementedError("Saving JPIV frames is not implemented yet (need single slicing)")
                
                # save frame
                self.write('jpiv', dataname=DATA_RAW, extention='.png', stacked=True)

        # close window
        cv2.destroyWindow('Live')

    def write(self, path:str=None, dataname:str=DATA_DRAWN, fps:float=30.0, extention:str='.jpg', stacked:bool=False) -> None:
        '''
        Write image or video

        Args:
            path (str) : path to save location (direcory or .avi file) (default=None)
            dataname (str) : data to write (default=DATA_DRAWN)
            fps (float) : video frames per second (default=30.0)
            extention (str) : image file extention (default='.jpg)
            stacked (bool) : stack reference frame on top of frame (default=False)

        Returns:
            None
        '''
        # get data
        if dataname == DATA_COMPUTED:
            imgs, _ = self._get_data(dataname=dataname, normalize=True)
        else:
            imgs, _ = self._get_data(dataname=dataname, normalize=False)

        # stack raw images image
        if (dataname == DATA_RAW and stacked):
            # unpack shape
            n, h, w, d = imgs.shape

            # prealocate stacked
            stacked = np.zeros((n-1, h * 2, w, d))

            # store frames
            stacked[:, h:, :, :] = imgs[1:]

            # store reference
            stacked[:, :h, :, :] = imgs[0]

            # overwrite old image
            imgs = stacked

        # pick current working directory and default to video if no path is given
        if not path:
            # get existing files and inital name guess
            files = os.listdir(os.getcwd())
            name = 'video.avi'

            # pick a valid name
            i = 1
            while name in files:
                name = f'video ({i}).avi'
                i += 1

            # save path
            path = name

        # get absolute path
        path = os.path.abspath(path)

        # check if path is for a video
        if os.path.splitext(path)[1] == EXTS_AVI:
            # build directory if needed
            direct = os.path.dirname(path)
            if not os.path.exists(direct):
                os.makedirs(direct)

            # get image shape
            size = (imgs.shape[2], imgs.shape[1])
            
            # open video writer
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            video_out = cv2.VideoWriter(path, fourcc, fps, size)

            # write video
            print("Writting Video")
            for frame in tqdm(imgs):
                video_out.write(frame)

            # release writer
            video_out.release()

        # check if path is for numpy
        elif os.path.splitext(path)[1] == EXTS_NUMPY:
            print("Writting Numpy File")
            # write as numpy
            with open(path, 'wb') as f:
                np.save(f, imgs)

        # check if path is for an image
        elif os.path.splitext(path)[1] in EXTS_IMAGE:
            # write first image
            print("Writting Single Image")
            cv2.imwrite(path, imgs[0])

        # check if path is for a directory
        elif '' in os.path.splitext(path)[1]:
            # build directory if needed
            if not os.path.exists(path):
                os.makedirs(path)

            # write images
            print("Writting Images")
            for i, frame in tqdm(enumerate(imgs)):
                cv2.imwrite(os.path.join(path, f'frame{i:04d}{extention}'), frame)

        else:
            raise ValueError(f'Expected path to be direcotry or .avi file but got {path}')