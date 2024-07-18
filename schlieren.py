import os
import cv2
import numpy as np
import tqdm as tqdm

# display methods
TOTAL = 'tot'
X = 'x'
Y = 'y'

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
        self._computed_X = None
        self._computed_Y = None
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
        # create empty computed data if needed (top preserve old data)
        if (type(self._computed) != np.ndarray) or (self._computed.shape != self._raw.shape):
            self._computed = np.zeros_like(self._raw)

        # setup slice
        if not stop:
            stop = len(self._raw)

        # slice
        data = self._raw[start:stop:step]

        # convert BGR to greyscale if needed
        isBGR = len(data.shape) > 3
        if isBGR:
            data = np.mean(data, axis=3)

        ########### COMPUTE ###########







        # convert greyscale to BGR if needed
        if isBGR:
            data = np.stack([data, data, data], axis=3)

        ########### COMPUTE ###########

        # store
        self._computed[start:stop:step] = data

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

        # placeholders
        ind = 0
        img = imgs[ind]

        # set window
        cv2.namedWindow(dataname)

        # loop
        while True:
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
                    img = imgs[ind]
            elif k == ord('d'):
                if ind < len(imgs) - 1:
                    ind += 1
                    img = imgs[ind]

            else:
                print('pressed:', k)

        # close window
        cv2.destroyWindow(dataname)

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