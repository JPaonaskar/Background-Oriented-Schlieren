'''
BOS TESTING
by Josha Paonaskar

Schlieren testing to validate batch methods

Resources:

'''
import cv2

import schlieren

if __name__ == '__main__':

    bos = schlieren.BOS()
    bos.read('D:\\Images\\06-16-24_testing\\Acquisition_001\\Y4-S2 Camera.avi')
    bos.compute(win_size=8, search_size=16, overlap=0, start=60)
    bos.draw(method=schlieren.DISP_MAG, thresh=5, interplolation=cv2.INTER_CUBIC, start=60)
    bos.write()
    #bos.live(win_size=32, search_size=64, overlap=16)