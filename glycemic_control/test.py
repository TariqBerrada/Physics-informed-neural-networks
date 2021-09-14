import numpy as np
import matplotlib.pyplot as plt

import joblib, sys, torch, os
sys.path.append('.')

from glycemic_control.model import GlycemicModel

data = joblib.load('data/glycemic_t.pt')['test']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GlycemicModel()
model.load_weights('./weights_temp.pth.tar')

_input = torch.from_numpy(data).float().to(device)
output = model(_input).detach().cpu().numpy()

plt.figure()
plt.scatter(data[:, 0], output[:, 0], s = 5)
plt.show()
