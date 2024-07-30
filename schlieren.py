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

# colors
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

# interplation
INTER_NEAREST = cv2.INTER_NEAREST
INTER_CUBIC = cv2.INTER_CUBIC


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

    # begin sprial (for number of step out from center) (Alternates from WS and WN)
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
                    # save coorinate
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
            self._raw = np.hstack([self._raw, np.array(data, dtype=np.uint8)])

        # write data
        else:
            self._raw = np.array(data, dtype=np.uint8)

    def _setup_compute(self, win_size:int, search_size:int, overlap:int, space:int, start:int, stop:int, step:int, pad:bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        '''
        Setup data and other values for computing

        Args:
            win_size (int) : search windows size
            search_size (int) : search size
            overlap (int) : overlap between windows
            space (int) : space between referance frame. None implies use start frame
            start (int) : starting frame
            stop (int) : ending frame (exclusive)
            step (int) : step between frames
            pad (bool) : pad edges

        Returns:
            raw_data (np.ndarray) : raw data values
            kernals (np.ndarray) : kernals
            win_x (np.ndarray) : x axis
            win_y (np.ndarray) : y axis
            n (int) : number of data points
            p (int) : padding size
        '''
        print('Setting Up Compute')
        # setup slice
        if not stop:
            stop = len(self._raw)
        if space and stop - space > start:
            stop = stop - space

        # slice
        raw_data = self._raw[start:stop:step]

        # unpack shape values
        n, h, w, d = raw_data.shape

        print('Converting to Grayscale')

        # convert BGR to greyscale if needed
        if d == 3:
            raw_data = vectorized_tools.grayscale(raw_data)

        # slice kernals
        if space:
            kernals = raw_data
            raw_data = self._raw[start+space:stop+space:step]

        # tile kernal
        else:
            kernals = np.expand_dims(raw_data[0], axis=0)
            kernals = np.tile(kernals, (n, 1, 1))

        print('Padding')

        # time offset data
        raw_data = raw_data[1:]
        kernals = kernals[:len(kernals)-1]

        # pad raw data
        p = (search_size - win_size) >> 1

        empty = np.zeros((n - 1, h + 2 * p, w + 2 * p), dtype=np.uint8)
        empty[:, p:h+p, p:w+p] = raw_data

        raw_data = empty
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
        return raw_data, kernals, win_x, win_y, n, p

    def compute(self, win_size:int=32, search_size:int=64, overlap:int=0, space:int=None, start:int=0, stop:int=None, step:int=1, pad:bool=False) -> None:
        '''
        Compute schlieren data

        Args:
            win_size (int) : search windows size (default=32)
            search_size (int) : search size (default=64)
            overlap (int) : overlap between windows (default=0)
            space (int) : space between referance frame. None implies use start frame (default=None)
            start (int) : starting frame (default=0)
            stop (int) : ending frame (exclusive) (default=None)
            step (int) : step between frames (default=1)
            pad (bool) : pad edges (default=False)

        Returns:
            None
        '''
        # get compute values
        raw_data, kernals, win_x, win_y, n, p = self._setup_compute(win_size, search_size, overlap, space, start, stop, step, pad)

        # create list of coordinate pairs
        win_coords = np.meshgrid(np.arange(len(win_x)), np.arange(len(win_y)))
        win_coords = np.vstack([win_coords[0].flatten(), win_coords[1].flatten()])
        win_coords = np.swapaxes(win_coords, 0, 1)

        # pre-allocate data (depth is u, v, length)
        data = np.zeros((n-1, len(win_y), len(win_x), 3))

        # task for threading
        def process(coord:np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
            # unpack output location
            row = coord[1]
            col = coord[0]

            # get window location
            win_row = win_y[row]
            win_col = win_x[col]

            # pull window
            win = kernals[:, win_row:win_row + win_size, win_col:win_col + win_size]

            # pull search area
            search = raw_data[:, win_row:win_row+win_size + 2 * p, win_col:win_col + win_size + 2 * p]

            # compute correlation and calcualte displacements
            corr = vectorized_tools.normxcorr2(search, win, mode='full')
            u, v = vectorized_tools.displacement(corr)

            # compute magnitude
            m = np.sqrt(np.square(u) + np.square(v))

            # output needed information to save data
            return row, col, u, v, m
        
        # threading for speed
        with ThreadPoolExecutor() as executor:
            futures = []

            # itterate though sections
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

    def compute_multi(self, win_size:int=32, search_size:int=64, particle_size:int=2, overlap:int=0, start:int=0, stop:int=None, step:int=1, pad:bool=False) -> None:
        '''
        Mulipass compute

        Args:
            win_size (int) : search windows size (default=32)
            search_size (int) : search size (default=64)
            particle_size (int) : particle size (default=2)
            overlap (int) : overlap between windows (default=0)
            space (int) : space between referance frame. None implies use start frame (default=None)
            start (int) : starting frame (default=0)
            stop (int) : ending frame (exclusive) (default=None)
            step (int) : step between frames (default=1)
            pad (bool) : pad edges (default=False)

        Returns:
            None
        '''
        pass

    def draw(self, method:str=DISP_MAG, thresh:float=5.0, alpha:float=0.6, colormap:int=cv2.COLORMAP_JET, interplolation:int=INTER_NEAREST, masked:float=None, start:int=0, stop:int=None, step:int=1) -> None:
        '''
        Draw computed data

        Args:
            method (str) : drawing method (default=DISP_MAG)
            thresh (float) : value maximum (defult=5.0)
            alpha (float) : blending between raw and computed (defult=0.6)
            colormap (int) : colormap (default=cv2.COLORMAP_JET)
            interplolation (int) : interplolation method (default=INTER_NEAREST)
            masked (float) : treat low displacements as a mask (default=None)
            start (int) : starting frame (default=0)
            stop (int) : ending frame (exclusive) (default=None)
            step (int) : step between frames (default=1)

        Returns:
            None
        '''
        # setup slice
        if not stop:
            stop = len(self._computed)

        # slice data
        drawn = self._raw[start:stop:step].copy()
        data = self._computed.copy()

        # time offset data
        drawn = drawn[1:]

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

        # remove zeros
        if masked:
            mask = np.bitwise_or(mask, data <= masked)

        # mask
        data[mask] = 0.0  

        # normalize data
        data = (data * 255 / thresh).astype(np.uint8)

        # draw images
        print('Drawing Frames')
        for i in tqdm(range(n)):
            point = data[i]
            raw = drawn[i]

            # apply colormap
            point = cv2.applyColorMap(point, colormap)

            # make blending image
            blend = mask[i].astype(np.uint8) * 255
            blend = np.stack([blend, blend, blend], axis=2)

            # resize
            blend = cv2.resize(blend, (w, h), interpolation=interplolation)
            point = cv2.resize(point, (w, h), interpolation=interplolation)

            # modify to be alpha
            blend = blend / 255.0 * alpha + (1 - alpha)

            # belnd and store
            drawn[i] = (raw * blend + point * (1 - blend)).astype(np.uint8)

        # store drawn data
        self._drawn = drawn

    def _get_data(self, dataname:str=DATA_DRAWN) -> np.ndarray:
        '''
        Get humanized data

        Args:
            dataname (str) : data to display (default=DATA_DRAWN)

        Returns:
            None
        '''
        # get images
        data = None
        if dataname == DATA_RAW:
            # get raw data
            data = self._raw
        elif dataname == DATA_COMPUTED:
            # get compuited data
            data = self._computed
            
            # shift unsigned 8-bit
            data = data.astype(np.float32)
            data = data - np.nanmin(data) # 0.0 - max
            data = data * 255.0 / np.nanmax(data) # 0.0 - 255.0
            data = data.astype(np.uint8) # 8bit

            # replace nan with zeros
            data = np.nan_to_num(data, nan=127)

        elif dataname == DATA_DRAWN:
            # get drawn data
            data = self._drawn
        else:
            ValueError(f'{dataname} in not a valid dataset')

        # output
        return data

    def display(self, dataname:str=DATA_DRAWN, font:int=cv2.FONT_HERSHEY_SIMPLEX, font_scale:float=0.5, font_color:tuple[int, int, int]=COLOR_WHITE, font_thickness:int=1, font_pad:int=8) -> None:
        '''
        Display drawn data

        Args:
            dataname (str) : data to display (default=DATA_DRAWN)
            font (int) : overlay font, None displays no text (default=cv2.FONT_HERSHEY_SIMPLEX)
            font_scale (float) : overlay font scale (default=0.5)
            font_color (tuple[int, int, int]) : overlay font color (default=COLOR_WHITE)
            font_thickness (int) : overlay font thickness (default=1)
            font_pad (int) : overlay font padding from edges (default=8)

        Returns:
            None
        '''
        # get data
        imgs = self._get_data(dataname=dataname)

        # placeholders
        ind = 0

        # set window
        cv2.namedWindow(dataname)

        # loop
        while True:
            # load new image
            img = imgs[ind]

            # resize
            img = cv2.resize(img, (512, 512), interpolation=INTER_NEAREST)

            # draw index
            if font != None:
                # get text size
                text = f'{ind+1} / {len(imgs)}'
                (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

                # put text
                org = (img.shape[1] - w - font_pad, img.shape[0] - h - font_pad)
                img = cv2.putText(img, text, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # draw frame
            cv2.imshow(dataname, img)

            # keys
            k = cv2.waitKey(0)

            # quit
            if (k == ord('q')) or (k == KEY_ESCAPE):
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
                print('pressed:', k, chr(k).strip())

        # close window
        cv2.destroyWindow(dataname)
    
    def _live_render_cell(self, win:np.ndarray, search:np.ndarray, method:str=DISP_MAG, thresh:float=5.0, alpha:float=0.6, colormap:int=cv2.COLORMAP_JET, masked:bool=False) -> np.ndarray:
        '''
        Compute and draw a cell

        Args:
            win (np.ndarray) : window to search for
            search (np.ndarray) : search area
            method (str) : drawing method (default=DISP_MAG)
            thresh (float) : value maximum (defult=5.0)
            alpha (float) : blending between raw and computed (defult=0.6)
            colormap (int) : colormap (default=cv2.COLORMAP_JET)
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

    def live(self, win_size:int=32, search_size:int=64, overlap:int=0, start:int=0, stop:int=None, step:int=1, pad:bool=False, save_win_size:int=32, save_search_size:int=64, save_overlap:int=0, method:str=DISP_MAG, thresh:float=5.0, alpha:float=0.6, colormap:int=cv2.COLORMAP_JET, interplolation=INTER_NEAREST, masked:float=None, font:int=cv2.FONT_HERSHEY_SIMPLEX, font_scale:float=0.5, font_color:tuple[int, int, int]=COLOR_WHITE, font_thickness:int=1, font_pad:int=8) -> None:
        '''
        Live computing and rendering

        Args:
            win_size (int) : search windows size (default=32)
            search_size (int) : search size (default=64)
            overlap (int) : overlap between windows (default=0)
            start (int) : starting frame (default=0)
            stop (int) : ending frame (exclusive) (default=None)
            step (int) : step between frames (default=1)
            pad (bool) : pad edges (default=False)

            save_win_size (int) : search windows size for saving (default=32)
            save_search_size (int) : search size for saving (default=64)
            save_overlap (int) : overlap between windows for saving (default=0)

            method (str) : drawing method (default=DISP_MAG)
            thresh (float) : value maximum (defult=5.0)
            alpha (float) : blending between raw and computed (defult=0.6)
            colormap (int) : colormap (default=cv2.COLORMAP_JET)
            interplolation (int) : interplolation method (default=INTER_NEAREST)
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
        raw_data, kernals, win_x, win_y, n, p = self._setup_compute(win_size, search_size, overlap, None, start, stop, step, pad)

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
                win = kernals[ind:ind+1, win_row:win_row + win_size, win_col:win_col + win_size]

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
            if (k == ord('q')) or (k == KEY_ESCAPE):
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
                i0 = start
                i1 = start + ind * step + 1
                di = ind * step

                # compute and draw frame
                self.compute(win_size=save_win_size, search_size=save_search_size, overlap=save_overlap, start=i0, stop=i1, step=di, pad=pad)
                self.draw(method=method, thresh=thresh, alpha=alpha, colormap=colormap, interplolation=interplolation, masked=masked, start=i0, stop=i1, step=di)
                
                # save frame
                self.write('results')
                print('Frame Saved')

        # close window
        cv2.destroyWindow('Live')

    def write(self, path:str=None, dataname:str=DATA_DRAWN, fps:float=30.0, start:int=0, stop:int=None, step:int=1, extention:str='.jpg') -> None:
        '''
        Write image or video

        Args:
            path (str) : path to save location (direcory or .avi file) (default=None)
            dataname (str) : data to write (default=DATA_DRAWN)
            fps (float) : video frames per second (default=30.0)
            start (int) : starting frame (default=0)
            stop (int) : ending frame (exclusive) (default=None)
            step (int) : step between frames (default=1)
            extention (str) : image file extention (default='.jpg)

        Returns:
            None
        '''
        # get data
        imgs = self._get_data(dataname=dataname)

        # slice
        imgs = imgs[start:stop:step]

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
        if '.avi' == os.path.splitext(path)[1]:
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