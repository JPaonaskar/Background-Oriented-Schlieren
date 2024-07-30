'''
BOS TESTING
by Josha Paonaskar

Schlieren testing to validate batch methods

Resources:

'''
import schlieren

if __name__ == '__main__':
    bos = schlieren.BOS()
    bos.read('Sample Data')

    # process data
    bos.compute(win_size=8, search_size=16, overlap=0)#, start=0, stop=26, step=25)
    #bos.draw(method=schlieren.DISP_MAG, thresh=3.2, alpha=0.5, masked=0, interplolation=schlieren.INTER_CUBIC, colormap=cv2.COLORMAP_INFERNO)

    # show user
    bos.display(schlieren.DATA_COMPUTED)