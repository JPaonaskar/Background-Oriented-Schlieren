'''
BOS TESTING
by Josha Paonaskar

Schlieren testing to validate batch methods

Resources:

'''

import schlieren

bos = schlieren.BOS()
bos.read('PIV Challange')
bos.compute(start=0, win_size=16, search_size=32, pad=False)
bos.draw(method=schlieren.DISP_MAG, thresh=5)
bos.display(schlieren.DATA_DRAWN)