import schlieren

bos = schlieren.BOS()
bos.read('PIV Challange')
bos.compute(start=0)
bos.display(schlieren.DATA_COMPUTED)