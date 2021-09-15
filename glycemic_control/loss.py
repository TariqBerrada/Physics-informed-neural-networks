import torch

criterion_0 = torch.nn.MSELoss(reduction = 'mean' )
criterion_b = torch.nn.MSELoss(reduction = 'mean' )

def l_0(model):

    N0 = 1
    
    t_0 = torch.zeros((N0, 1), dtype = torch.float32).to(model.device)
    G_tar = torch.linspace(13.5, 14, steps = N0).to(model.device)
    X_tar = torch.zeros(N0).to(model.device)
    I_tar = .5*torch.ones(N0).to(model.device)

    target = torch.stack((G_tar, I_tar, X_tar), dim = 1).to(model.device)

    t_0 = t_0.to(model.device)

    pred = model(t_0)
    target.to(model.device)
    l_0 = criterion_0(pred, target)
    return l_0
    
def l_b(model, conditions):
    
    preds = model(conditions[0][None][None])
    Gt_p = model.G_dt(conditions[0][None][None])
    Xt_p = model.X_dt(conditions[0][None][None])
    It_p = model.I_dt(conditions[0][None][None])

    all = torch.cat((preds, Gt_p, Xt_p, It_p), dim = 1)

    # print('limit conditions', all.shape, conditions[1:][None].shape)
    mse_b = criterion_b(all, conditions[1:][None])
    return mse_b