'''
BOS TESTING
by Josha Paonaskar

Schlieren testing to validate batch methods

Resources:

'''

import schlieren

bos = schlieren.BOS()
bos.read('PIV Challange')
bos.compute(start=0)
bos.display(schlieren.DATA_COMPUTED)