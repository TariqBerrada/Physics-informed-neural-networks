import torch, joblib
import numpy as np

import sys
sys.path.append('.')

from torch.utils.data import DataLoader

from glycemic_control.model import GlycemicModel
from tools.data_processing import GlycemicDatasetClass
from tools.trainer import train_glycemic

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GlycemicModel().train().to(device)
model.load_weights('weights/glycemic_control/init.pth.tar')
batch_size = 20000

Nt = 50000
Nf = 20000

t_f = np.linspace(0, 720, Nt)[None].T
np.random.shuffle(t_f)

t_f_train = t_f[:Nf, ...]
t_f_test = t_f[Nf:, ...]

joblib.dump({'train':t_f_train, 'test': t_f_test}, './data/glycemic_t.pt')

train_set = GlycemicDatasetClass(t_f_train)
test_set = GlycemicDatasetClass(t_f_test)

train_loader = DataLoader(train_set, batch_size = batch_size)
test_loader = DataLoader(test_set, batch_size = batch_size)

optimizer = torch.optim.LBFGS(model.parameters(), lr = 1, max_eval = 2000, history_size = 50, max_iter = 2000, line_search_fn = 'strong_wolfe', tolerance_grad=1e-4, tolerance_change= np.finfo(float).eps)
# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, amsgrad = True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', .2, 500)

# Train model for 50 epochs.
train_loss, val_loss, lr_list = train_glycemic(model, train_loader, test_loader, optimizer, scheduler, 5, weights_dir = './weights/glycemic_control/init.pth.tar', type_ = 'LBFGS')

# Check out final learning plot.
plt.subplot(121)
plt.plot(train_loss)
plt.subplot(122)
plt.plot(val_loss)
plt.savefig('./figures/learning_glycemic.jpg')
plt.show()