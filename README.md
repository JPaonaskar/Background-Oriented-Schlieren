# Background-Oriented-Schlieren

## 1. Intoduction

## 2. Examples

## 3. Algorithm

## 4. Documentation

### 4.1. Quick Start

### 4.2. Module: schlieren

#### 4.2.1. Classes

**`class BOS()`**

```
    read(
        path : str,
        append : bool = False
    ) -> None
```

```
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

```
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

```
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

```
    _get_data(
        dataname : str = DATA_DRAWN
    ) -> np.ndarray
```

```
    display(
        dataname : str = DATA_DRAWN,
        font : int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale : float = 0.5,
        font_color : tuple[int, int, int] = COLOR_WHITE,
        font_thickness : int = 1,
        font_pad : int = 8
    ) -> None
```

```
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

```
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

```
    write(
        path : str = None,
        dataname : str = DATA_DRAWN,
        fps : float = 30.0,
        start : int = 0,
        stop : int = None,
        step : int = 1
    ) -> None
```

#### 4.2.2. Functions

```
_spiral_coords(
    x : int,
    y : int
) -> coords : np.ndarray
```

#### 4.2.3. Constants

**Display Methods**
```
DISP_MAG = 'mag'
DISP_X = 'x'
DISP_Y = 'y'
```

**Datasets**
```
DATA_RAW = 'raw'
DATA_COMPUTED = 'computed'
DATA_DRAWN = 'drawn'
```

**File Extentions**
```
EXTS_IMAGE = ['.tif', '.jpg', '.png']
EXTS_VIDEO = ['.avi']
```

**Keys**
```
KEY_BACKSPACE = 8
KEY_TAB = 9
KEY_ENTER = 13
KEY_ESCAPE = 27
KEY_SPACE = 32
```

**Colors**
```
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
```

### 4.3. Module: vectorized_tools

#### 4.3.1. Functions

```
conv2D(
    images : np.ndarray,
    kernals : np.ndarray,
    mode : str = 'full'
) -> out : np.ndarray
```

```
grayscale(
    images : np.ndarray
) -> out : np.ndarray
```

```
batch_subtract(
    images : np.ndarray,
    values : np.ndarray
) -> out : np.ndarray
```

```
batch_multiply(
    images : np.ndarray,
    values : np.ndarray
) -> out : np.ndarray
```

```
normxcorr2(
    images : np.ndarray,
    keranls : np.ndarray,
    mode : str = 'full'
) -> corr : np.ndarray
```

```
gaussian(
    s : np.ndarray
) -> dr : float
```

```
displacement(
    corr : np.ndarray,
    precision : type
) -> tuple[
    x : np.ndarray,
    y : np.ndarray
]
```

#### 4.3.2. Constants

**convolution modes**
```
CONV_MODE_FULL = 'full'
CONV_MODE_VALID = 'valid'
```

## Style Guide
https://google.github.io/styleguide/pyguide.html