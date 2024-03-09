from PIL import Image
from matplotlib import cm
import numpy as np
import os.path

max_val = 100.0

for it in list(range(1000,100000,1000)):
    fname = "heat_" + str(it) + ".csv"
    if not os.path.isfile(fname):
        break

    f = open(fname, 'r')
    lines = f.readlines()
    f.close()

    print("making image for " + fname)
    data = []

    for i in list(range(len(lines))):
        nums = lines[i].split(",")
        data.append([float(n) for n in nums[0:-1]])


    for i in list(range(len(data))):
        for j in list(range(len(data[i]))):
            data[i][j] = data[i][j] / max_val

    im = Image.fromarray(np.uint8(cm.plasma(data)*255))

    imgfname = "heat_" + str(it) + ".png"
    im.save(imgfname)
