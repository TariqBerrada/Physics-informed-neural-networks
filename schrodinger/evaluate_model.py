import numpy as np

import tqdm, torch, joblib, sys
sys.path.append('.')

from tools.model import NN
from tools.utils import make_gif
import matplotlib.pyplot as plt

data = joblib.load('data/schrodinger_pts.pt')
data_test = data['test']
data_train = data['train']

print(f'test data : {data_test.shape} | train data : {data_train.shape}')

timesteps = list(set(data_test[:, 0]))

batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NN(2, 2, 60, 5, batch_size = batch_size).to(device)
model.load_weights('weights/lbfgs.pth.tar')

for _id, timestep in tqdm.tqdm(enumerate(timesteps)):

    t_ids_test = np.where(data_test[:, 0] == timestep)[0]
    t_ids_train = np.where(data_train[:, 0] == timestep)[0]

    outputs = []
    x_inputs = []

    t_data = data_test[t_ids_test]

    for i in range(0, t_data.shape[0], batch_size):

        batch = t_data[i:i+batch_size, ...]
        
        if not batch.shape[0] == batch_size:
            batch = t_data[t_data.shape[0] - batch_size:t_data.shape[0], ...]

        _input = torch.tensor(batch[:, [3, 2]], dtype = torch.float32, device = model.device)

        x = _input[:, 0]
        t = _input[:, 1]

        output = model(x, t)
        outputs.append(output.detach().cpu().numpy())
        x_inputs.append(batch[:, 3])

    outputs = np.array(outputs).reshape((-1, 2))
    x_inputs = np.array(x_inputs).reshape(-1)
    
    plt.figure()

    plt.scatter(data_train[t_ids_train, 3], data_train[t_ids_train, 4], s = 5, c = 'b')
    plt.scatter(data_train[t_ids_train, 3], data_train[t_ids_train, 5], s = 5, c = 'r')

    plt.scatter(x_inputs, outputs[:, 0], marker = '*', s = 5)
    plt.scatter(x_inputs, outputs[:, 1], marker = '*', s = 5)
    
    plt.title(f'timestep = {timestep}')
    plt.legend(['train_u', 'train_v', 'pred_u', 'pred_v'], loc = 'lower left')
    
    plt.xlim(-5, 5)
    plt.ylim(-4.5, 3)

    plt.savefig('figures/test_model/img_%.3d.jpg'%_id)
    plt.close()

make_gif('./test_model/', './figures/solution_test.gif')
