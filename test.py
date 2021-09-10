import numpy as np

import tqdm, torch, joblib

from tools.data_processing import DatasetClass
from tools.model import NN
from tools.utils import make_gif
import matplotlib.pyplot as plt

data = joblib.load('data/schrodinger_pts.pt')
data_test = data['test']
data_train = data['train']
print(data_test.shape, data_train.shape)


timesteps = list(set(data_test[:, 0]))

for _id, timestep in tqdm.tqdm(enumerate(timesteps)):


    t_ids_test = np.where(data_test[:, 0] == timestep)[0]
    # print(t_ids_test)
    print('number points of points in timestep', timestep, len(t_ids_test))

    t_ids_train = np.where(data_train[:, 0] == timestep)[0]

    # print(t_ids_train)

    
    batch_size = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NN(2, 2, 50, 5, batch_size = batch_size).to(device)
    model.load_weights('weights/basic.pth.tar')
    model.eval()


    outputs = []
    t_data = data_test[t_ids_test]

    for i in range(0, data_test.shape[0], batch_size):
        # print(i)
        #print('sizes of the data', data_test.shape, i, i+batch_size)
        # batch = data_test[i:i+batch_size, :]
        batch = t_data[i:i+batch_size, :]
        # print('shape of batch', batch.shape)
        if batch.shape[0] == batch_size:
            # print('shape of batch', batch.shape, i, i+batch_size)
            # print('input', batch[:, [3, 2]])

            _input = torch.tensor(batch[:, [3, 2]], dtype = torch.float32, device = model.device)
            # print('input', _input.shape)
            output = model(_input)
            # print('ouput', output.shape)
            outputs.append(output.detach().cpu().numpy())
            # print('output', output)

    outputs = np.array(outputs).reshape((-1, 2))
    # print('oo', outputs.shape)

    plt.figure()

    plt.scatter(data_train[t_ids_train, 4], data_train[t_ids_train, 5], s = 5, c = 'b')
    plt.scatter(data_test[t_ids_test, 4], data_test[t_ids_test, 5], s = 5, c = 'r')
    
    plt.scatter(outputs[:, 0], outputs[:, 1], s = 5, c = 'green')
    plt.title(f'timestep = {timestep}')
    
    plt.xlim(-4, 2.5)
    plt.ylim(-4, 3)

    plt.savefig('test_model/img_%d.jpg'%_id)
    plt.close()
make_gif('./test_model/', 'solution_test.gif')
    # plt.show()

# for i, x in enumerate(range(data.shape[0])):
#     print(i, data[x, 2], data[x, 1], data[x, 0])

    