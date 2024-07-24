'''
BOS TESTING
by Josha Paonaskar

Schlieren testing to validate batch methods

Resources:

'''

import schlieren

bos = schlieren.BOS()
bos.read('Sample Data')
bos.compute(start=0, win_size=8, search_size=16, pad=False)
bos.draw(method=schlieren.DISP_MAG, thresh=5)
bos.display(schlieren.DATA_DRAWN)