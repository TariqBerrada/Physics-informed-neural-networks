import numpy as np
import matplotlib.pyplot as plt

import joblib, sys, torch, os
sys.path.append('.')

from glycemic_control.model import GlycemicModel

data = joblib.load('data/glycemic_t.pt')['test']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GlycemicModel()
model.load_weights('./weights/glycemic_control/init.pth.tar')

_input = torch.from_numpy(data).float().to(device)
output_1 = model(_input).detach().cpu().numpy()

_ids = np.argsort(data[:, 0])

plt.figure()
plt.plot(data[_ids, 0], output_1[_ids, 0])


del model
model = GlycemicModel()
model.load_weights('./weights/glycemic_control/transition_3.0.pth.tar')
output_2 = model(_input).detach().cpu().numpy()

plt.plot(data[_ids, 0], output_2[_ids, 0])


plt.legend(['initial', 'after transition'])
plt.show()
