# Background-Oriented-Schlieren

Images and sample data to come!

## 1. Intoduction

## 2. Examples

## 3. Algorithms

## 4. Documentation

### 4.1 Quick Start

### 4.1.1 BOS Class

`class BOS()` manages data and processing as the backgound oriented schlieren.
Generally this is all that is needed to perform BOS.

```python
import schlieren

bos = schlieren.BOS()
```


### 4.1.2 Reading Data

Raw data can be read using `.read(...)` method.
This method can read a single image:

```python
bos.read('image.jpg')
```

.avi video files:

```python
bos.read('video.avi')
```

and folders of images:

```python
bos.read('frames')
```

`.read(...)` by default appends any read data to the end of what is already stored.
To overwrite existing data set the argument `append=False`.

> [!TIP]
> Any stored data can be visualized using the `.display(...)` method.
> The data (`dataname`) can be set to `DATA_RAW`, `DATA_COMPUTED`, or `DATA_DRAWN`.


### 4.1.3 Computing Displacements

Displacements are computed using the `.compute(...)` method.
The `win_size` controls the size of the window and `search_size` controls the size of the area to search for the window

```python
bos.compute(win_size=8, search_size=16)
```

> [!NOTE]
> `search_size` must be larger than `win_size` and is usually best when the differance is larger than the expected displacements

> [!TIP]
> `overlap` can be used for super resolution


### 4.1.4 Drawing Displacements

Displacements are drawn using the `.draw(...)` method.

```python
bos.draw()
```

> [!TIP]
> Draw has three `methods`: `DISP_X`, `DISP_Y`, and `DISP_MAG` (default).

The data can be clipped using the `thresh` as the maximum value and `masked` as the minumum.
To modify the blending between the background and the data set `alpha` to between `0.0` (no bg) and `1.0` (only bg)

```python
bos.draw(thresh=5.0, alpha=0.6, masked=0.5)
```

### 4.1.5 Writing Data

The `.write(...)` method writes data as images to a folder:

```python
bos.write('frames')
```

or a video:

```python
bos.write('video.avi')
```

> [!TIP]
> By default a video is outputed to the current working directory.

> [!TIP]
> The frames per second can be set using `fps`.

### 4.1.6 Live View

Data can be processed and rendered live using the `.live(...)` method.
This method contains all the arguments as compute, draw, and display with the addition of compute parameters for saving.

```python
bos.live()
```

> [!IMPORTANT]
> Use `a` and `d` to take large steps between framse and `,` and `.` for single frames.
> Use `s` to save the current frame

> [!TIP]
> The save resolution can be set seperately with `save_...` arguments

### 4.2. Module: schlieren

### 4.2.1. Classes

**`class BOS()`**

```python
def read(
    path : str,
    append : bool = False
) -> None
```

```python
    def read_jpiv(
        path : str,
        stacked : bool = True
) -> None
```

```python
def _setup_compute(
    win_size : int,
    search_size : int,
    space : int,
    start : int,
    stop : int,
    step : int,
    pad : bool
) -> tuple[
    raw_data : np.ndarray,
    kernals : np.ndarray,
    win_x : np.ndarray,
    win_y : np.ndarray,
    n : int,
    p : int
]
```

```python
def compute(
    win_size : int = 32,
    search_size : int = 64,
    overlap : int = 0,
    space : int = None,
    start : int = 0,
    stop : int = None,
    step : int = 1,
    pad : bool = False
) -> None
```

```python
def compute_multi(
    win_size : int = 32,
    search_size : int = 64,
    particle_size : int = 2,
    overlap : int = 0,
    start : int = 0,
    stop : int = None,
    step : int = 1,
    pad : bool = False
) -> None
```

```python
def draw(
    method : str = DISP_MAG,
    thresh : float = 5.0,
    alpha : float = 0.6,
    colormap : int = cv2.COLORMAP_JET,
    interplolation : int = INTER_NEAREST,
    masked : bool = False,
    start : int = 0,
    stop : int = None,
    step : int = 1
) -> None
```

```python
def _get_data(
    dataname : str = DATA_DRAWN
) -> np.ndarray
```

```python
def display(
    dataname : str = DATA_DRAWN,
    font : int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale : float = 0.5,
    font_color : tuple[int, int, int] = COLOR_WHITE,
    font_thickness : int = 1,
    font_pad : int = 8
) -> None
```

```python
def _live_render_cell(
    win : np.ndarray,
    search : np.ndarray,
    method : str = DISP_MAG,
    thresh : float = 5.0,
    alpha : float = 0.6,
    colormap = cv2.COLORMAP_JET,
    masked : bool = False
) -> cell : np.ndarray
```

```python
def live(
    win_size : int = 32,
    search_size : int = 64,
    overlap : int = 0,
    start : int = 0,
    stop : int = None,
    step : int = 1,
    pad : bool = False,

    save_win_size : int = 32
    save_search_size : int = 64
    save_overlap : int = 0

    method : str = DISP_MAG,
    thresh : float = 5.0,
    alpha : float = 0.6,
    colormap : int = cv2.COLORMAP_JET,
    interplolation : int = INTER_NEAREST,
    masked : float = None,

    font : int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale : float = 0.5,
    font_color : tuple[int, int, int] = COLOR_WHITE,
    font_thickness : int = 1,
    font_pad : int = 8
) -> None
```

```python
def write(
    path : str = None,
    dataname : str = DATA_DRAWN,
    fps : float = 30.0,
    start : int = 0,
    stop : int = None,
    step : int = 1,
    extention : str = '.jpg',
    stacked : bool = False
) -> None
```

### 4.2.2. Functions

```python
def _spiral_coords(
    x : int,
    y : int
) -> coords : np.ndarray
```

### 4.2.3. Constants

**Display Methods**
```python
DISP_MAG = 'mag'
DISP_X = 'x'
DISP_Y = 'y'
```

**Datasets**
```python
DATA_RAW = 'raw'
DATA_COMPUTED = 'computed'
DATA_DRAWN = 'drawn'
```

**File Extentions**
```python
EXTS_IMAGE = ['.tif', '.jpg', '.png']
EXTS_VIDEO = ['.avi']
```

**Keys**
```python
KEY_BACKSPACE = 8
KEY_TAB = 9
KEY_ENTER = 13
KEY_ESCAPE = 27
KEY_SPACE = 32
```

**Colors**
```python
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
```

**Interpolation**
```python
INTER_NEAREST = cv2.INTER_NEAREST
INTER_CUBIC = cv2.INTER_CUBIC
```

### 4.3. Module: vectorized_tools

### 4.3.1. Functions

```python
def conv2D(
    images : np.ndarray,
    kernals : np.ndarray,
    mode : str = 'full'
) -> out : np.ndarray
```

```python
def grayscale(
    images : np.ndarray
) -> out : np.ndarray
```

```python
def batch_subtract(
    images : np.ndarray,
    values : np.ndarray
) -> out : np.ndarray
```

```python
def batch_multiply(
    images : np.ndarray,
    values : np.ndarray
) -> out : np.ndarray
```

```python
def normxcorr2(
    images : np.ndarray,
    keranls : np.ndarray,
    mode : str = 'full'
) -> corr : np.ndarray
```

```python
def gaussian(
    s : np.ndarray
) -> dr : float
```

```python
def displacement(
    corr : np.ndarray,
    precision : type
) -> tuple[
    x : np.ndarray,
    y : np.ndarray
]
```

### 4.3.2. Constants

**convolution modes**
```python
CONV_MODE_FULL = 'full'
CONV_MODE_VALID = 'valid'
```

### 4.4. Module: vectorized_tools

### 4.4.1. Functions

```python
def noise(
    shape : tuple,
    scale : float = 0.1
) -> np.ndarray
```

```python
def synthetic_dataset(
    batch_size : int,
    win_size : int = 32,
    search_size : int = 64,
    noise_scale : float = 0.1
) -> np.ndarray
```

```python
def batch_size_test(
    batch_sizes : list[int],
    test_window : float = 5.0,
    win_size : int = 32,
    search_size : int = 64,
    noise_scale : float = 0.1,
    show : bool = True
) -> None
```

### 4.4.2. Constants

```python
PATTERN = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])
```

## Bugs and Tasks

1. Noise give very large displacements. Values are removed in vector_tools.py -> displacements() but results are still not ideal
2. Write function is clucky and needs to be streamlined
3. Add support for non-stacked JPIV images (or remove feature)
4. Implement multi-pass
5. Add blur/smoothing when reading images
6. Reduce memory usage
7. Add recent changes to README
8. Remove slicing in compute / draw / write

## Target Style
https://google.github.io/styleguide/pyguide.html