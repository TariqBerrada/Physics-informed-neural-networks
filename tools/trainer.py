import torch
import tqdm

import sys
sys.path.append('.')

import numpy as np
# from tools.losses import loss_0, loss_b, loss_f
from glycemic_control.loss import l_0, l_b
import matplotlib.pyplot as plt

import sys, os
sys.path.append('.')

def validate_glycemic(model, dataloader, limit_conditions = None):
    model.eval()
    device = model.device
    running_loss = 0.0
    # print('lml__________', len(dataloader.dataset), dataloader.batch_size)
    total = int(len(dataloader.dataset)/dataloader.batch_size)
    
    for i, data in tqdm.tqdm(enumerate(dataloader), total =total):

        t_f = data['t_f'].float().to(device)
        if limit_conditions is None:
            u_type = 1
        else:
            if limit_conditions[0] >= 6:
                u_type = 1
            else:
                u_type = 2
        mse_f = (model.eq_1(t_f)**2 + model.eq_2(t_f)**2 + model.eq_3(t_f, u_type = u_type)**2).mean()

        _loss = mse_f

        running_loss += _loss.item()
    val_loss = running_loss/total
    return val_loss

def fit_glycemic(model, dataloader, optimizer, scheduler, limit_conditions = None, type_ = 'LBFGS'):
    model.train()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    running_loss = 0.0
    
    total = int(len(dataloader.dataset)/dataloader.batch_size)

    for i, data in tqdm.tqdm(enumerate(dataloader), total = total):
        optimizer.zero_grad()

        # data_0 = data['data_0'].float().to(device)
        t_f = data['t_f'].float().to(device)

        if type_ == 'LBFGS':
            def closure():
                optimizer.zero_grad()
                mse_0 = l_0(model)

                if limit_conditions is None:
                    u_type = 1
                else:
                    if limit_conditions[0] >= 6:
                        u_type = 1
                    else:
                        u_type = 2

                mse_f = (model.eq_1(t_f)**2 + model.eq_2(t_f)**2 + model.eq_3(t_f, u_type = u_type)**2).mean()

                if limit_conditions is not None:
                    mse_b = l_b(model, limit_conditions)
                    mse_0 = 0
                else:
                    mse_b = 0

                # print(mse_0, mse_f.item(), mse_b.item())
                _loss = mse_0 + mse_f + mse_b

                # print(f'l_0 : {l_0} | l_b : {l_b} | l_f {l_f}')
                # print('lll', mse_0, mse_f)

                _loss.backward()
                return _loss

            optimizer.step(closure)

            # calculate loss again for monitoring.
            mse_0 = l_0(model)
            if limit_conditions is None:
                u_type = 1
            else:
                if limit_conditions[0] >= 6:
                    u_type = 1
                else:
                    u_type = 2
            mse_f = (model.eq_1(t_f)**2 + model.eq_2(t_f)**2 + model.eq_3(t_f, u_type = u_type)**2).mean()

            if limit_conditions is not None:
                mse_b = l_b(model, limit_conditions)
                mse_0 = 0
            else:
                mse_b = 0

            _loss = mse_0 + mse_f + mse_b

            running_loss += _loss.item()

        else:

            mse_0 = l_0(model)

            if limit_conditions is None:
                u_type = 1
            else:
                if limit_conditions[0] >= 6:
                    u_type = 1
                else:
                    u_type = 2
            mse_f = (model.eq_1(t_f)**2 + model.eq_2(t_f)**2 + model.eq_3(t_f, u_type = u_type)**2).mean()

            if limit_conditions is not None:
                mse_b = l_b(model, limit_conditions)
                mse_0 = 0
            else:
                mse_b = 0

            _loss = mse_0 + mse_f + mse_b
    
            running_loss += _loss.item()
            _loss.backward()
            optimizer.step()

    train_loss = running_loss/total
    scheduler.step(train_loss)
    return train_loss

def train_glycemic(model, train_loader, val_loader, optimizer, scheduler, n_epochs, limit_conditions = None, weights_dir = './weights/basic.pth.tar', type_ = 'LBFGS'):
    train_loss = []
    val_loss = []
    lr_list = []

    min_loss = np.inf

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_epoch_loss = fit_glycemic(model, train_loader, optimizer, scheduler, limit_conditions = limit_conditions, type_ = type_)
        val_epoch_loss = validate_glycemic(model, val_loader, limit_conditions = limit_conditions)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        lr_list.append(optimizer.param_groups[0]['lr'])
        print('L_train : ', train_epoch_loss)
        print('L_val : ', val_epoch_loss)
        print(' - lr : ', optimizer.param_groups[0]['lr'])

        if val_epoch_loss < min_loss:
            if limit_conditions is not None:
                c = limit_conditions.detach().cpu().numpy()
            else:
                c = None

            save_dict = {
                'epoch' : epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'state_dict' : model.state_dict(),
                'conditions' : c
            }
            torch.save(save_dict, weights_dir)
            min_loss = val_epoch_loss

        if epoch%2 == 0:
            save_dict = {
                'epoch' : epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'state_dict' : model.state_dict(),
                'conditions' : c
            }
            torch.save(save_dict, weights_dir[:-8]+'_ckpt.pth.tar')

            plt.subplot(121)
            plt.plot(train_loss)
            plt.title('train')
            plt.subplot(122)
            plt.plot(val_loss)
            plt.title('test')
            plt.savefig('./figures/learning_glycemic.jpg')
            plt.close()

        
    return train_loss, val_loss, lr_list

def fit(model, dataloader, optimizer, scheduler, type_ = 'LBFGS'):
    model.train()

    device = model.device
    running_loss = 0.0

    
    
    total = int(len(dataloader.dataset)/dataloader.batch_size)

    for i, data in tqdm.tqdm(enumerate(dataloader), total = total):
        optimizer.zero_grad()

        data_0 = data['data_0'].float().to(device)
        data_b = data['data_b'].float().to(device)
        data_f = data['data_f'].float().to(device)

        if type_ == 'LBFGS':
            def closure():
                optimizer.zero_grad()
                l_0 = loss_0(data_0, model)
                l_b = loss_b(data_b, model)
                l_f = loss_f(data_f, model, None)

                _loss = l_0 + l_b + l_f  

                # print(f'l_0 : {l_0} | l_b : {l_b} | l_f {l_f}')

                _loss.backward()
                return _loss

            optimizer.step(closure)

            # calculate loss again for monitoring.
            l_0 = loss_0(data_0, model)
            l_b = loss_b(data_b, model)
            l_f = loss_f(data_f, model, None)

            _loss = l_0 + l_b + l_f
            running_loss += _loss.item()

        else:

            l_0 = loss_0(data_0, model)
            l_b = loss_b(data_b, model)
            l_f = loss_f(data_f, model, None)

            _loss = l_0 + l_b + l_f
    
            running_loss += _loss.item()
            _loss.backward()
            optimizer.step()

    train_loss = running_loss/total
    scheduler.step(train_loss)
    return train_loss

def validate(model, dataloader):
    model.eval()
    device = model.device
    running_loss = 0.0
    total = int(len(dataloader.dataset)/dataloader.batch_size)
    
    for i, data in tqdm.tqdm(enumerate(dataloader), total =total):

        data_0 = data['data_0'].float().to(device)
        data_b = data['data_b'].float().to(device)
        data_f = data['data_f'].float().to(device)
        
        l_0 = loss_0(data_0, model)
        l_b = loss_b(data_b, model)
        l_f = loss_f(data_f, model, None)

        _loss = l_0 + l_b + l_f

        running_loss += _loss.item()
    val_loss = running_loss/total
    return val_loss

def train(model, train_loader, val_loader, optimizer, scheduler, n_epochs, weights_dir = './weights/basic.pth.tar', type_ = 'LBFGS'):
    train_loss = []
    val_loss = []
    lr_list = []

    min_loss = np.inf

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_epoch_loss = fit(model, train_loader, optimizer, scheduler, type_ = type_)
        val_epoch_loss = validate(model, val_loader)

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        lr_list.append(optimizer.param_groups[0]['lr'])
        print('L_train : ', train_epoch_loss)
        print('L_val : ', val_epoch_loss)
        print(' - lr : ', optimizer.param_groups[0]['lr'])

        if val_epoch_loss < min_loss:
            save_dict = {
                'epoch' : epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'state_dict' : model.state_dict()
            }
            torch.save(save_dict, weights_dir)
            min_loss = val_epoch_loss

        if epoch%2 == 0:
            save_dict = {
                'epoch' : epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'state_dict' : model.state_dict()
            }
            torch.save(save_dict, weights_dir[:-8]+'_ckpt.pth.tar')

            plt.subplot(121)
            plt.plot(train_loss)
            plt.title('train')
            plt.subplot(122)
            plt.plot(val_loss)
            plt.title('test')
            plt.savefig('./figures/learning.jpg')
            plt.close()

        
    return train_loss, val_loss, lr_list

def predict(model, data):
    model.eval()
    device = model.device
    loader = torch.from_numpy(data).float().to(device)
    out = model(loader).detach().cpu().numpy()
    return out