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

x = data[_ids, 0]
sp = np.argmin((x - 181)**2)
# sp2 = np.argmin((x - 27)**2)
print(sp)

plt.figure()
plt.plot(data[_ids, 0][:sp], output_1[_ids, 0][:sp], 'b')
plt.plot(data[_ids, 0][sp:], output_1[_ids, 0][sp:],'b--')


del model
model = GlycemicModel()
model.load_weights('./weights/glycemic_control/transition_181.0.pth.tar')
output_2 = model(_input).detach().cpu().numpy()

plt.plot(data[_ids, 0][:sp], output_2[_ids, 0][:sp], 'r--')
plt.plot(data[_ids, 0][sp:], output_2[_ids, 0][sp:], 'r')

# del model
# model = GlycemicModel()
# model.load_weights('./weights/glycemic_control/transition_39.0.pth.tar')
# output_3 = model(_input).detach().cpu().numpy()

# plt.plot(data[_ids, 0], output_3[_ids, 0])
plt.ylim([-5, 16])
plt.xlabel('time $(min)$')
plt.ylabel('$G (mmol.L^{-1})$')
plt.legend(['$t<t_1$ : $u = u_1$','$t \geq t_1$ : $u = u_1$', '$t<t_1$ : $u = u_2$','$t \geq t_1$ : $u = u_2$'])
plt.savefig('./figure.jpg')
plt.show()
