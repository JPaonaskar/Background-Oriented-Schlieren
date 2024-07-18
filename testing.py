import schlieren

bos = schlieren.BOS()
bos.read('PIV Challange')
print(bos.raw.shape)