from PIL import Image
from matplotlib import cm
import numpy as np

max_iter = 1000.0

f = open("mandelbrot.csv", 'r')
lines = f.readlines()
f.close()

data = []

for i in list(range(len(lines))):
    nums = lines[i].split(",")
    data.append([int(n) for n in nums[0:-1]])

for i in list(range(len(data))):
    for j in list(range(len(data[i]))):
        data[i][j] = data[i][j] / max_iter

im = Image.fromarray(np.uint8(cm.plasma(data)*255))

im.save("mandel.png")