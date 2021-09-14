import torch, joblib
import numpy as np

import sys
sys.path.append('.')

from glycemic_control.model import GlycemicModel
from glycemic_control.loss import l_b
from glycemic_control.utils import train

def traverse_time(t_init, model, limit_values = None, all_preds= []):
    t_remaining = torch.arange(t_init, 751, step = 1).float().to(model.device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if limit_values is not None:
        # initialize a new model and train it.    
        model = GlycemicModel().train().to(device)
        train(model, limit_values)

    preds = model(t_remaining[None].T)

    for i in range(preds.shape[0] - 1):
        if (preds[i, 0] - 6)*(preds[i+1, 0] - 6) <= 0:
            t_new = t_remaining[i]
            grads=  model.G_dt(t_remaining[None].T)[i, 0], model.I_dt(t_remaining[None].T)[i, 0], model.X_dt(t_remaining[None].T)[i, 0]
            limit_values = torch.stack([t_new, preds[i, 0], preds[i, 1], preds[i, 2], *grads]).detach()
            
            # return traverse_time(t_new, model, limit_values = limit_values)
            ######## loss = l_b(model, limit_values)
            del model
            model = GlycemicModel().train().to(device)
            all_preds.append(preds)
            
            return traverse_time(t_new + 1, model, limit_values = limit_values, all_preds = all_preds)

            
    print('done traversing !')
    return all_preds
    



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_init = GlycemicModel().to(device)
    model_init.load_weights('weights/glycemic_control/init.pth.tar')

    all_preds = traverse_time(0, model_init)
    joblib.dump(all_preds, './data/preds_glycemic.pt')