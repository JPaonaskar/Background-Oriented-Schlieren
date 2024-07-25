'''
BOS TESTING
by Josha Paonaskar

Schlieren testing to validate batch methods

Resources:

'''
import cv2

import schlieren

bos = schlieren.BOS()
bos.read('Sample Data')
bos.compute(start=0, win_size=8, search_size=16, pad=False)
bos.draw(method=schlieren.DISP_MAG, thresh=5, masked=0.5, interplolation=cv2.INTER_CUBIC)
bos.display(schlieren.DATA_DRAWN)
#bos.write()
#bos.live(win_size=16, search_size=32, masked=0.4)