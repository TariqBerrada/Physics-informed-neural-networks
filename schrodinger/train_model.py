import numpy as np

import torch, joblib, sys
sys.path.append('.')

from tools.trainer import train
# from tools.utils import make_gif
from tools.data_processing import DatasetClass
from torch.utils.data import DataLoader
from tools.model import NN

# from scipy.io import loadmat

import matplotlib.pyplot as plt

data = joblib.load('data/schrodinger_pts.pt')

points_f, points_f_test, points_b, points_0 = data['train'], data['test'], data['b'], data['0'] 

# Transform data into torch loader.
f_dataset = DatasetClass(points_0, points_b, points_f)
f_dataset_test = DatasetClass(points_0, points_b, points_f_test)

batch_size = 20000

loader = DataLoader(f_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(f_dataset_test, batch_size = batch_size, shuffle = True)

# Instanciate model with 5 fully connected layers and 60 neurons per layer.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NN(2, 2, 60, 5, batch_size=batch_size).train().to(device)
# model.load_weights('weights/lbfgs_ckpt.pth.tar')

# Define optimizer.
optimizer = torch.optim.LBFGS(model.parameters(), lr = 1.0, max_eval = 500, history_size = 50, max_iter = 500, line_search_fn = 'strong_wolfe', tolerance_grad=1e-5, tolerance_change=1.0 * np.finfo(float).eps)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad = False)

# Define learning rate scheluder.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', .2, 500)

# Train model for 50 epochs.
train_loss, val_loss, lr_list = train(model, loader, test_loader, optimizer, scheduler, 50, weights_dir = './weights/schrodinger_60_5.pth.tar', type_ = 'LBFGS')

# Check out final learning plot.
plt.subplot(121)
plt.plot(train_loss)
plt.subplot(122)
plt.plot(val_loss)
plt.savefig('./figures/learning.jpg')
plt.show()