import torch

criterion_0 = torch.nn.MSELoss(reduction = 'mean' )
criterion_b = torch.nn.MSELoss(reduction = 'mean' )
criterion_appr = torch.nn.MSELoss(reduction = 'mean')


def l_0(model):
    """Initial conditions loss.

    Args:
        model ([type]): [description]

    Returns:
        [type]: [description]
    """

    N0 = 1
    
    t_0 = torch.zeros((N0, 1), dtype = torch.float32).to(model.device)
    G_tar = torch.linspace(13.8, 14, steps = N0).to(model.device)
    X_tar = torch.zeros(N0).to(model.device)
    I_tar = .5*torch.ones(N0).to(model.device)

    target = torch.stack((G_tar, I_tar, X_tar), dim = 1).to(model.device)

    t_0 = t_0.to(model.device)

    pred = model(t_0)
    target.to(model.device)
    l_0 = criterion_0(pred, target)
    return l_0

def l_appr(model, state = 1):
    """Approximation loss.
    collocation loss of type (f(xi) - yi)**2

    Args:
        model ([type]): [description]
        state (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    # Hardcoded collocation values to use as limit conditions.
    target = torch.tensor([0.40827130997541566, -0.12952662566531847, -1.7896865370716464], dtype = torch.float32).to(model.device)[None].T
    t =  torch.tensor([50., 175., 100], dtype = torch.float32).to(model.device)[None].T

    if state == 2:
        target = torch.tensor([5.639932487903715], dtype = torch.float32).to(model.device)[None].T
        t =  torch.tensor([500], dtype = torch.float32).to(model.device)[None].T


    l_appr = criterion_appr(target, model(t)[:, [0]])

    return l_appr


def l_b(model, conditions):
    """Gradient continuity loss.

    Args:
        model ([type]): [description]
        conditions ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    preds = model(conditions[0][None][None])

    Gt_p = model.G_dt(conditions[0][None][None])
    Xt_p = model.X_dt(conditions[0][None][None])
    It_p = model.I_dt(conditions[0][None][None])

    all = torch.cat((preds, Gt_p, Xt_p, It_p), dim = 1)

    mse_b = 10*criterion_b(all, conditions[1:][None])
    return mse_b