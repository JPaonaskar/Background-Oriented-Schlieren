# Background-Oriented-Schlieren

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
> Any stored data can be visualized using the `.display(...)` method in `BOS()`.
> The data (`dataname`) can be set to `DATA_RAW`, `DATA_COMPUTED`, or `DATA_DRAWN`.


### 4.1.3 Computing Displacements

Displacements are computed using the `.compute(...)` method.
The `win_size` controls the size of the window to search for while `search_size` controls the size of the area to search for the window

```python
bos.compute(win_size=8, search_size=16)
```

> [!NOTE]
> `search_size` must be larger than `win_size` and is usually best around 2x

The selection of frames can be changed with the `start`, `stop` (exclusive), and `step`.


### 4.1.4 Drawing Displacements

Displacements are drawn using the `.draw(...)` method.

```python
bos.draw()
```

> [!TIP]
> Draw has three `methods`: `DISP_X`, `DISP_Y`, and `DISP_MAG` (default).

The data can be clipped using the `thresh` as the maximum value and `masked` as the minumum.
Use `colormap` to set the colormap to any OpenCV color map.
To modify the blending between the background and the data set `alpha` to between `0.0` (no bg) and `1.0` (only bg)

```python
bos.draw(thresh=5.0, alpha-0.6, masked=0.5, colormap=cv2.COLORMAP_JET)
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
> The frames per section is set using `fps`.

### 4.1.6 Live View

Data can be processed and rendered live using the `.live(...)` method.
This method contains all the arguments as compute, draw, and display with the exception of interpolation.

```python
bos.live()
```

> [!IMPORTANT]
> Use `a` and `d` to take large steps between framse and `,` and `.` for single frames.
> Use `s` to save the drawn frame

### 4.2. Module: schlieren

### 4.2.1. Classes

**`class BOS()`**

```python
read(
    path : str,
    append : bool = False
) -> None
```

```python
_setup_compute(
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
compute(
    win_size : int = 32,
    search_size : int = 64,
    space : int = None,
    start : int = 0,
    stop : int = None,
    step : int = 1,
    pad : bool = False
) -> None
```

```python
draw(
    method : str = DISP_MAG,
    thresh : float = 5.0,
    alpha : float = 0.6,
    colormap : int = cv2.COLORMAP_JET,
    interplolation : int = cv2.INTER_NEAREST,
    masked : bool = False,
    start : int = 0,
    stop : int = None,
    step : int = 1
) -> None
```

```python
_get_data(
    dataname : str = DATA_DRAWN
) -> np.ndarray
```

```python
display(
    dataname : str = DATA_DRAWN,
    font : int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale : float = 0.5,
    font_color : tuple[int, int, int] = COLOR_WHITE,
    font_thickness : int = 1,
    font_pad : int = 8
) -> None
```

```python
_live_render_cell(
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
live(
    win_size : int = 32,
    search_size : int = 64,
    start : int = 0,
    stop : int = None,
    step : int = 1,
    pad : bool = False,

    method : str = DISP_MAG,
    thresh : float = 5.0,
    alpha : float = 0.6,
    colormap : int = cv2.COLORMAP_JET,
    masked : bool = False,

    font : int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale : float = 0.5,
    font_color : tuple[int, int, int] = COLOR_WHITE,
    font_thickness : int = 1,
    font_pad : int = 8
) -> None
```

```python
write(
    path : str = None,
    dataname : str = DATA_DRAWN,
    fps : float = 30.0,
    start : int = 0,
    stop : int = None,
    step : int = 1
) -> None
```

### 4.2.2. Functions

```python
_spiral_coords(
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

### 4.3. Module: vectorized_tools

### 4.3.1. Functions

```python
conv2D(
    images : np.ndarray,
    kernals : np.ndarray,
    mode : str = 'full'
) -> out : np.ndarray
```

```python
grayscale(
    images : np.ndarray
) -> out : np.ndarray
```

```python
batch_subtract(
    images : np.ndarray,
    values : np.ndarray
) -> out : np.ndarray
```

```python
batch_multiply(
    images : np.ndarray,
    values : np.ndarray
) -> out : np.ndarray
```

```python
normxcorr2(
    images : np.ndarray,
    keranls : np.ndarray,
    mode : str = 'full'
) -> corr : np.ndarray
```

```python
gaussian(
    s : np.ndarray
) -> dr : float
```

```python
displacement(
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

## Target Style
https://google.github.io/styleguide/pyguide.html