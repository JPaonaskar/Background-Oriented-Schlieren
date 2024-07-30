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
    bos.read_jpiv("jpiv\\frame00200.jvc")

    # process data
    bos.draw(method=schlieren.DISP_MAG, thresh=5, alpha=0.5, masked=0, interplolation=schlieren.INTER_CUBIC, colormap=cv2.COLORMAP_INFERNO)

    # show user
    bos.display(schlieren.DATA_DRAWN)

    # write test
    bos.write('jpiv', dataname=schlieren.DATA_RAW, stacked=True)