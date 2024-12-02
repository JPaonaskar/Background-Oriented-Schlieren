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
    bos.read(r'PIV Challange')

    # show raw
    #bos.display(dataname=schlieren.DATA_RAW, normalize=False)

    # blur
    bos.gaussianBlur()

    # show blurred
    #bos.display(dataname=schlieren.DATA_RAW, normalize=False)

    # split data into pairs
    bos.split(start=0, step=1, method=schlieren.PAIR_CASCADE)

    # show pairs
    #bos.display(dataname=schlieren.DATA_SPLIT)

    # live draw
    #bos.live(win_size=12, search_size=24, save_win_size=8, save_search_size=16, save_overlap=6, thresh=2.0, alpha=0.5, masked=0, colormap=schlieren.COLORMAP_INFERNO)

    # compute live
    bos.compute(win_size=16, search_size=32, overlap=0)

    # show computed
    #bos.display(dataname=schlieren.DATA_COMPUTED)

    # save computed
    #bos.write('computed.npy', dataname=schlieren.DATA_COMPUTED)

    # read computed
    #bos.read('computed.npy', computed=True)

    # show read computed
    #bos.display(dataname=schlieren.DATA_COMPUTED)

    # process data
    #bos.draw(method=schlieren.DISP_VECTOR, thresh=2.0, alpha=0.8, interpolation=schlieren.INTER_CUBIC, colormap=schlieren.COLORMAP_COMPUTED)

    # show drawn
    #bos.display(dataname=schlieren.DATA_DRAWN)

    # draw quiver
    bos.quiver(thresh=5, alpha=0.3, colormap=schlieren.COLORMAP_COMPUTED, thickness=10, interpolation=schlieren.INTER_NEAREST, scale=10)

    # save as img
    bos.write('quiver')

    # show drawn
    bos.display(dataname=schlieren.DATA_DRAWN)

    '''
    for name, cmap in [('computed', schlieren.COLORMAP_COMPUTED), ('ocean', schlieren.COLORMAP_OCEAN), ('bone', schlieren.COLORMAP_BONE),('hot', schlieren.COLORMAP_HOT), ('inferno', schlieren.COLORMAP_INFERNO), ('viridis', schlieren.COLORMAP_VIRIDIS)]:
        # process data
        bos.draw(method=schlieren.DISP_MAG, thresh=3.0, alpha=1.0, interpolation=schlieren.INTER_CUBIC, colormap=cmap)

        # show drawn
        #bos.display(dataname=schlieren.DATA_DRAWN)
    '''
    # save data
    #bos.write(f'test.avi', dataname=schlieren.DATA_DRAWN, fps=3)
    