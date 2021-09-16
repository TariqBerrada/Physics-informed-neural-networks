import torch, joblib
import numpy as np

import sys
sys.path.append('.')

from torch.utils.data import DataLoader

from glycemic_control.model import GlycemicModel
from tools.data_processing import GlycemicDatasetClass
from tools.trainer import train_glycemic

import matplotlib.pyplot as plt

def train(model, conditions):

    batch_size = 20000

    Nt = 50000
    Nf = 20000
    
    data = joblib.load('./data/glycemic_t.pt')

    t_f_train = data['train']

    t_f_train = t_f_train[np.where(t_f_train > conditions[0].detach().cpu().numpy())][None].T
    batch_size = t_f_train.shape[0]

    t_f_test = data['test']

    train_set = GlycemicDatasetClass(t_f_train)
    test_set = GlycemicDatasetClass(t_f_test)

    train_loader = DataLoader(train_set, batch_size = batch_size)
    test_loader = DataLoader(test_set, batch_size = batch_size)

    optimizer = torch.optim.LBFGS(model.parameters(), lr = .1, max_eval = 2000, history_size = 50, max_iter = 2000, line_search_fn = 'strong_wolfe', tolerance_grad=1e-4, tolerance_change=1.0 * np.finfo(float).eps)
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, amsgrad = True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', .2, 200)

    if conditions is not None:
        print('training model at timestep : ', conditions[0] , ' | G = ', conditions[1])

    # Train model for 50 epochs.
    train_loss, val_loss, lr_list = train_glycemic(model, train_loader, test_loader, optimizer, scheduler, 200, limit_conditions = conditions, weights_dir = f'./weights/glycemic_control/transition_{conditions[0]}.pth.tar', type_ = 'LBFGS')
    
    # Check out final learning plot.
    plt.subplot(121)
    plt.plot(train_loss)
    plt.subplot(122)
    plt.plot(val_loss)
    plt.savefig(f'./figures/learning_{conditions[0]}.jpg')
    

    return model