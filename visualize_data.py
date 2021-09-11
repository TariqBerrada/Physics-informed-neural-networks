import numpy as np
from scipy.io import loadmat
from tools.utils import make_gif

import matplotlib.pyplot as plt

xx = loadmat('./data/NLS.mat')

x = xx['x'][0]
t = xx['tt'][0]
f = xx['uu']

print(x.shape, t.shape, f.shape)

u = f.real
v = f.imag

print(u.shape, v.shape)

for t in range(f.shape[1]):
    plt.figure()
    plt.plot(x, u[:, t])
    plt.plot(x, v[:, t])
    plt.xlim(-5, 5)
    plt.ylim(-4, 2.5)
    plt.legend(['real', 'imaginary'])
    plt.savefig('temp/img_%.3d.jpg'%t)
    plt.close()

make_gif('temp', './sch_data.gif')
