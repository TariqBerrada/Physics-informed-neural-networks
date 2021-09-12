import numpy as np

import tqdm, torch, joblib

from scipy.io import loadmat
from tools.utils import make_gif
from tools.data_processing import DatasetClass
from torch.utils.data import DataLoader

from tools.model import NN

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

print(u_tar.shape, '-------------')
print(u_tar)
u_coords, v_coords, h_coords = u_tar[x_ids, 0], v_tar[x_ids, 0], h_tar[x_ids, 0]

points_0 = np.stack((x_ids, x_coords, u_coords, v_coords, h_coords)).T
print('0', x_ids.shape, x_coords.shape, u_coords.shape, v_coords.shape, h_coords.shape, x_ids.shape, points_0.shape)

# 2. Nb = 50 randomly sampled collocation points (temporal) for the periodic boundaries.
Nb = 50

t_ids = np.random.choice(range(len(t)), Nb, replace = False)

t_coords = t[t_ids]

points_b = np.stack((t_ids, t_coords)).T
print('b', points_b.shape)

# 3. Nf = 20.000 pts to enforce the differential equation conformity.
Nf = 20000

f_ids = np.random.choice(range(np.product(Exact.shape)), Nf, replace = False)
ids_test = np.array([ j for j in range(np.product(Exact.shape)) if not j in f_ids])

t_ids, x_ids = (f_ids//Exact.shape[0], f_ids%Exact.shape[0])
t_ids_test, x_ids_test = (ids_test//Exact.shape[0], ids_test%Exact.shape[0])

t_values, x_values = t[t_ids], x[x_ids]
u_values, v_values, f_values = u_tar[x_ids, t_ids], v_tar[x_ids, t_ids], h_tar[x_ids, t_ids]

t_test, x_test, u_test, v_test, h_test = t[t_ids_test], x[x_ids_test], u_tar[x_ids_test, t_ids_test], v_tar[x_ids_test, t_ids_test], h_tar[x_ids_test, t_ids_test]

points_f = np.stack((t_ids, x_ids, t_values, x_values, u_values, v_values, f_values)).T
points_f_test = np.stack((t_ids_test, x_ids_test, t_test, x_test, u_test, v_test, h_test)).T
print('f', t_values.shape, x_values.shape, u_values.shape, v_values.shape, f_values.shape, points_f.shape)

joblib.dump({'train' : points_f, 'test' : points_f_test}, 'data/schrodinger_pts.pt')
# Transform data into torch loader.

f_dataset = DatasetClass(points_0, points_b, points_f)
f_dataset_test = DatasetClass(points_0, points_b, points_f_test)

batch_size = 20000

loader = DataLoader(f_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(f_dataset_test, batch_size = batch_size, shuffle = True)
# for batch in loader:
#     for k, v in batch.items():
#         print(k, ' -> ', v.shape)

from tools.trainer import train

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NN(2, 2, 60, 5, batch_size=batch_size).train().to(device)
# model.load_weights('weights/lbfgs_ckpt.pth.tar')

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad = False)
optimizer = torch.optim.LBFGS(model.parameters(), lr = 1.0, max_eval = 500, history_size = 50, max_iter = 500, line_search_fn = 'strong_wolfe', tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', .2, 500)

train_loss, val_loss, lr_list = train(model, loader, test_loader, optimizer, scheduler, 50, weights_dir = './weights/lbfgs.pth.tar', type_ = 'LBFGS')

plt.subplot(121)
plt.plot(train_loss)
plt.subplot(122)
plt.plot(val_loss)
plt.savefig('./learning.jpg')
plt.show()