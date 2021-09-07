import numpy as np

import tqdm

from scipy.io import loadmat
from tools.utils import make_gif
import matplotlib.pyplot as plt

data = loadmat('data/NLS.mat')

# Load data to know what it is
for k, v in data.items():
    print(k, ' --> ', v.shape if isinstance(v, np.ndarray) else type(v))

Exact = data['uu']
t = data['tt'].flatten()[:,None]
x = data['x'].flatten()[:,None]

print(f'x {x.shape} | t {t.shape}')

u_tar = np.real(Exact)
v_tar = np.imag(Exact)
h_tar = np.sqrt(u_tar**2 + v_tar**2)

print(u_tar.shape, v_tar.shape, h_tar.shape)

X, T = np.meshgrid(x,t)

print(f'X {X.shape} | T {T.shape}')

xmin, xmax = u_tar.min(), u_tar.max()
ymin, ymax = v_tar.min(), v_tar.max()

for i in tqdm.tqdm(range(u_tar.shape[1])):
    plt.figure()
    plt.plot(u_tar[:, i], v_tar[:, i])


    plt.title(f'timestep = {i}')
    plt.xlabel('Real(h)')
    plt.ylabel('Imaginary(h)')
    plt.xlim(xmin - .1, xmax + .1)
    plt.ylim(ymin - .1, ymax + .1)
    plt.savefig('./temp/img_%.3d.jpg'%i)
    plt.close()

make_gif('./temp', './shrodinger.gif')
