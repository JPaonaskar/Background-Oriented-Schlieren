# Background-Oriented-Schlieren

## 1. Intoduction

## 2. Examples

## 3. Algorithm

## 4. Documentation

### 4.1. Module: schlieren

#### 4.1.1. Classes

```
class BOS(

) 
```

#### 4.1.2. Functions

```
_spiral_coords(
    x : int,
    y : int
) -> np.ndarray
```

#### 4.1.3. Constants

### 4.2. Module: vectorized_tools

#### 4.1.1. Classes

#### 4.1.2. Functions

```
conv2D(
    images : np.ndarray,
    kernals : np.ndarray,
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
    keranls : np.ndarray
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

#### 4.1.3. Constants

## Style Guide
https://google.github.io/styleguide/pyguide.html