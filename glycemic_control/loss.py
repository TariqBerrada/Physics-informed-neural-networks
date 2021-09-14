import torch

criterion_0 = torch.nn.MSELoss(reduction = 'mean' )

def l_0(model):

    N0 = 50
    
    t_0 = torch.zeros((N0, 1), dtype = torch.float32).to(model.device)
    G_tar = torch.linspace(13, 14, steps = N0).to(model.device)
    X_tar = torch.zeros(N0).to(model.device)
    I_tar = .5*torch.ones(N0).to(model.device)

    target = torch.stack((G_tar, I_tar, X_tar), dim = 1).to(model.device)

    t_0 = t_0.to(model.device)

    pred = model(t_0)
    target.to(model.device)
    l_0 = criterion_0(pred, target)
    return l_0
    