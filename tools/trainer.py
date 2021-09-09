import torch
import tqdm

import numpy as np
from tools.losses import loss_0, loss_b, loss_f

import sys, os
sys.path.append('.')

def fit(model, dataloader, optimizer, scheduler):
    model.train()

    device = model.device
    running_loss = 0.0

    for i, data in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)//dataloader.batch_size):
        optimizer.zero_grad()

        # _input = data['data'].float().to(model.device)

        data_0 = data['data_0'].float().to(model.device)
        data_b = data['data_b'].float().to(model.device)
        data_f = data['data_f'].float().to(model.device)

        # input_0 = torch.cat((data_0[:, 1], torch.zeros_like(data_0[:, 1])))
        
        l_0 = loss_0(data_0, model)
        l_b = loss_b(data_b, model)
        l_f = loss_f(data_f, model, None)

        _loss = l_0 + l_b + l_f

        # prediction_0 = model(data_0)

        # prediction = model(_input)

        
        # ground_truth = data['prediction'].float().to(model.device) - mean_temperature
        # # _loss = loss(prediction, ground_truth)
        # _loss = aleatoric_loss(prediction, ground_truth)

        running_loss += _loss.item()
        _loss.backward()
        optimizer.step()

    train_loss = running_loss/len(dataloader.dataset)
    scheduler.step(train_loss)
    return train_loss

def validate(model, dataloader):
    model.eval()
    device = model.device
    running_loss = 0.0
    for i, data in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)//dataloader.batch_size):
        # _input = data['data'].float().to(device)
        # with torch.no_grad():
        #     prediction = model(_input)
        #     mean_temperature = torch.tile(data['data'].mean(dim = 1)[:, 0], (3, 7, 1)).transpose(0, -1).float().to(device)
        #     ground_truth = data['prediction'].float().to(model.device)
        #     # _loss = loss(prediction, ground_truth)
        #     _loss = aleatoric_loss(prediction, ground_truth)
        #     running_loss += _loss.item()
        with torch.no_grad():
            data_0 = data['data_0'].float().to(model.device)
            data_b = data['data_b'].float().to(model.device)
            data_f = data['data_f'].float().to(model.device)

            # input_0 = torch.cat((data_0[:, 1], torch.zeros_like(data_0[:, 1])))
            
            l_0 = loss_0(data_0, model)
            l_b = loss_b(data_b, model)
            l_f = loss_f(data_f, model, None)

            _loss = l_0 + l_b + l_f

            running_loss += _loss.item()
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss

def train(model, train_loader, val_loader, optimizer, scheduler, n_epochs, weights_dir = './weights/basic.pth.tar'):
    train_loss = []
    val_loss = []
    lr_list = []

    min_loss = np.inf

    for epoch in tqdm.tqdm(range(n_epochs)):
        train_epoch_loss = fit(model, train_loader, optimizer, scheduler)
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
        if epoch%100 == 0:
            save_dict = {
                'epoch' : epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'state_dict' : model.state_dict()
            }
            torch.save(save_dict, weights_dir[:-8]+'_ckpt.pth.tar')

        
    return train_loss, val_loss, lr_list

def predict(model, data):
    model.eval()
    device = model.device
    loader = torch.from_numpy(data).float().to(device)
    out = model(loader).detach().cpu().numpy()
    return out