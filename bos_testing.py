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
    #bos.compute(win_size=8, search_size=16, overlap=4, start=60, stop=101, step=10)
    #bos.draw(method=schlieren.DISP_MAG, thresh=5, interplolation=cv2.INTER_CUBIC, start=60, stop=101, step=10)
    #bos.display()
    #bos.write()
    bos.live(start=5, win_size=16, search_size=32, overlap=0, save_win_size=12, save_search_size=24, save_overlap=10, interplolation=schlieren.INTER_CUBIC)