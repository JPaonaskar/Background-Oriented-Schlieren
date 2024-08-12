import numpy as np
import cv2
import os

width = 1600
scale = 1
i = 2

arr = np.random.random((width, int(width * 11/8.5)))
arr = (2 * arr).astype(np.uint8) * 255
arr = cv2.resize(arr, (scale * int(width * 11/8.5), scale * width), interpolation=cv2.INTER_NEAREST)

cv2.imshow('frame', arr)
cv2.waitKey(0)
path = os.path.join(os.getcwd(), f'frame{width}-{i}.jpg')
cv2.imwrite(path, arr)
print(path)