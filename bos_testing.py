import schlieren

bos = schlieren.BOS()
bos.read('PIV Challange')
bos.compute(start=0, stop=5)
bos.display(schlieren.DATA_COMPUTED)