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
t = data['tt'].flatten()
x = data['x'].flatten()

print(f'x {x.shape} | t {t.shape}')

u_tar = np.real(Exact)
v_tar = np.imag(Exact)
h_tar = np.sqrt(u_tar**2 + v_tar**2)

print(u_tar.shape, v_tar.shape, h_tar.shape)



# xmin, xmax = u_tar.min(), u_tar.max()
# ymin, ymax = v_tar.min(), v_tar.max()

# for i in tqdm.tqdm(range(u_tar.shape[1])):
#     plt.figure()
#     plt.plot(u_tar[:, i], v_tar[:, i])


#     plt.title(f'timestep = {i}')
#     plt.xlabel('Real(h)')
#     plt.ylabel('Imaginary(h)')
#     plt.xlim(xmin - .1, xmax + .1)
#     plt.ylim(ymin - .1, ymax + .1)
#     plt.savefig('./temp/img_%.3d.jpg'%i)
#     plt.close()

# make_gif('./temp', './shrodinger.gif')

# subdivide data into training ad testing and bounday data.

# 1. N0 = 50 data pts for h(0, x)
N0 = 50

x_ids = np.random.choice(range(len(x)), N0, replace = False)

x_coords = x[x_ids]
u_coords, v_coords, h_coords = u_tar[x_ids, 0], v_tar[x_ids, 0], h_tar[x_ids, 0]

points_0 = np.stack((x_coords, u_coords, v_coords, h_coords))
print('0', x_ids.shape, x_coords.shape, u_coords.shape, v_coords.shape, h_coords.shape, x_ids.shape, points_0.shape)

# 2. Nb = 50 randomly sampled collocation points (temporal) for the periodic boundaries.
Nb = 50

t_ids = np.random.choice(range(len(t)), Nb, replace = False)

t_coords = t[t_ids]

points_b = np.stack((t_ids, t_coords))
print('b', points_b.shape)

# 3. Nf = 20.000 pts to enforce the differential equation conformity.
Nf = 20000

f_ids = np.random.choice(range(np.product(Exact.shape)), Nf, replace = False)
t_ids, x_ids = (f_ids//Exact.shape[0], f_ids%Exact.shape[0])

t_values, x_values = t[t_ids], x[x_ids]
u_values, v_values, f_values = u_tar[x_ids, t_ids], v_tar[x_ids, t_ids], h_tar[x_ids, t_ids]

f_points = np.stack((t_ids, x_ids, t_values, x_values, u_values, v_values, f_values))
print('f', t_values.shape, x_values.shape, u_values.shape, v_values.shape, f_values.shape, f_points.shape)
