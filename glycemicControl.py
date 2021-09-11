import numpy as np

import tqdm, torch, joblib

from scipy.io import loadmat
from tools.utils import make_gif
from tools.data_processing import DatasetClass
from torch.utils.data import DataLoader

from tools.model import NN

import matplotlib.pyplot as plt

"""
    File used to simulate glycemic control
    It contains functions inspired by the ODEs describes at https://ths.rwth-aachen.de/research/projects/hypro/glycemic-control/
"""

def P(t):
    """
        Returns the rate of infusion of exogenous glucose (mmol/L)
        Parameters:
            -t : current time (minutes)
    """
    return 0.5*np.exp(-0.05*t)

def u1(G):
    """
        Returns the rate of infusion of exogenous insulin (U/h) following strategy I
        Parameters:
            -G: difference of plasma glucose concentration (mmol/L)
    """
    if G < 4:
        return 0.5
    elif G > 8:
        return 2.5
    else:
        return 0.5*G - 1.5

def u2(G):
    """
        Returns the rate of infusion of exogenous insulin (U/h) following strategy II
        Parameters:
            -G: difference of plasma glucose concentration (mmol/L)
    """
    if G < 2:
        return 0.5
    elif G > 12:
        return 2.5
    else:
        return 0.2*G + 0.1

def u3(G):
    """
        Returns the rate of infusion of exogenous insulin (U/h) following strategy III
        Parameters:
            -G: difference of plasma glucose concentration (mmol/L)
    """
    if G >= 6 :
        return G * (0.41 - 0.0094*G)
    else:
        return 0.007533 * (1 + 0.22*G)

def u(G, strategy_number):
    """
        Returns the rate of infusion of exogenous insulin (U/h) following wanted strategy
        Parameters:
            -G: difference of plasma glucose concentration (mmol/L)
            -strategy_number : wanted strategy to use
    """
    if strategy_number == 1:
        return u1(G)
    elif strategy_number == 2:
        return u2(G)
    elif strategy_number == 3:
        return u3(G)
    else:
        # Use third strategy by default
        return u3(G)

def MSE_0(g, g_0, x, x_0, i, i_0):
    """
        Compute the loss associated to initial conditions.
        Parameters:
            -g (float) : estimated initial condition of G by the network 
            -g_0 (float) : real initial condition of G
            -x (float) : estimated initial condition of X by the network 
            -x_0 (float) : real initial condition of X
            -i (float) : estimated initial condition of I by the network 
            -i_0 (float) : real initial condition of I
    """
    return (g - g_0)**2 + (x - x_0)**2 + (i - i_0)**2

def MSE_1(G, G_point, X, T, Nf, params):
    """
        Compute the loss associated to the first equation.
        Parameters:
            -G (float array-like) : estimated values of G by the NN
            -G_point (float array-like) : estimated values of the derivative of G by the NN
            -X (float array-like) : estimated values of X by the NN
            -T (float array-like) : associated time for each value of G, G_point, ... used for the loss
            -Nf (integer) : number of points to use for the loss
            -params (dict) : parameters of the equations
    """
    loss = 0
    for i in range(Nf):
        loss += (G_point[i] + params['p1']*G[i] + X[i]*(G[i] + params['Gb']) - P(T[i]))**2
    loss /= Nf
    return loss

def MSE_2(X_point, X, I, Nf, params):
    """
        Compute the loss associated to the second equation.
        Parameters:
            -X_point : estimated values of the derivative of X by the NN
            -X : estimated values of X by the NN
            -I : estimated values of I by the NN
            -Nf : number of points to use for the loss
            -params : parameters of the equations
    """
    loss = 0
    for i in range(Nf):
        loss += (X_point[i] + params['p2']*X[i] - params['p3']*I[i])**2
    loss /= Nf
    return loss

def MSE_3(G, I_point, I, Nf, params):
    """
        Compute the loss associated to the first equation.
        Parameters:
            -G : estimated values of G by the NN
            -I_point : estimated values of the derivative of I by the NN
            -I : estimated values of I by the NN
            -Nf : number of points to use for the loss
            -params : parameters of the equations
    """
    loss = 0
    for i in range(Nf):
        # The 60 factor corresponds to a conversion from /h to /min
        loss += (I_point[i] + params['n']*(I[i] + params['Ib']) - u(G, params['strategy_number'])/(60*params['V1']))**2
    loss /= Nf
    return loss

def MSE(g, g_0, x, x_0, i, i_0, G, G_point, X, X_point, I, I_point, T, Nf):
    """
        Compute the total loss for the NN
        Parameters:
            -g : estimated initial condition of G by the network 
            -g_0 : real initial condition of G
            -x : estimated initial condition of X by the network 
            -x_0 : real initial condition of X
            -i : estimated initial condition of I by the network 
            -i_0 : real initial condition of I
            -G : estimated values of G by the NN
            -G_point : estimated values of the derivative of G by the NN
            -X : estimated values of X by the NN
            -X_point : estimated values of the derivative of X by the NN
            -I : estimated values of I by the NN
            -I_point : estimated values of the derivative of I by the NN
            -T : associated time for each value of G, G_point, ... used for the loss
            -Nf : number of points to use for the loss
    """
    params = {'p1' : 0,
              'p2' : 0.025,
              'p3' : 0.013,
              'Gb' : 4.5,
              'Ib' : 0.015,
              'n' : 5/54,
              'V1' : 12,
              'strategy_number' : 3}
    MSE0 = MSE_0(g, g_0, x, x_0, i, i_0)
    MSE1 = MSE_1(G, G_point, X, T, Nf, params)
    MSE2 = MSE_2(X_point, X, I, Nf, params)
    MSE3 = MSE_3(G, I_point, I, Nf, params)
    MSE_tot = MSE0 + MSE1 + MSE2 + MSE3
    return MSE_tot


if __name__ == "__main__":
    data = loadmat('data/NLS.mat')

    # Load data to know what it is
    for k, v in data.items():
        print(k, ' --> ', v.shape if isinstance(v, np.ndarray) else type(v))

    # Exact corresponds to the interesting data
    ExactG = data['gg']
    ExactX = data['xx']
    ExactI = data['ii']
    # t corresponds to the times possible for the data
    t = data['tt'].flatten()

    # Display the number of x coordinates and t coordinates in the dataset
    print(f't {t.shape}')

    # Necessary for complex data, useless in our case 
    # u_tar = np.real(Exact)
    # v_tar = np.imag(Exact)
    # h_tar = np.sqrt(u_tar**2 + v_tar**2)

    gmin, gmax = ExactG.min(), ExactG.max()
    xmin, xmax = ExactX.min(), ExactX.max()
    imin, imax = ExactI.min(), ExactI.max()

    # subdivide data into training ad testing and bounday data.

    # 1. N0 = 1 data pts for g(0)
    N0 = 1

    points_0 = [ExactG[0], ExactX[0], ExactI[0]]

    # 2. Nf = 20.000 pts to enforce the differential equation conformity.
    Nf = 20000

    f_ids = np.random.choice(range(np.product(ExactG.shape)), Nf, replace = False)

    ids_test = np.array([ j for j in range(np.product(ExactG.shape)) if not j in f_ids])

    t_ids, x_ids = (f_ids//ExactG.shape[0], f_ids%ExactG.shape[0])

    # t_ids_test, x_ids_test = (ids_test//Exact.shape[0], ids_test%Exact.shape[0])

    # t_values, x_values = t[t_ids], x[x_ids]
    # u_values, v_values, f_values = u_tar[x_ids, t_ids], v_tar[x_ids, t_ids], h_tar[x_ids, t_ids]

    # t_test, x_test, u_test, v_test, h_test = t[t_ids_test], x[x_ids_test], u_tar[x_ids_test, t_ids_test], v_tar[x_ids_test, t_ids_test], h_tar[x_ids_test, t_ids_test]

    # points_f = np.stack((t_ids, x_ids, t_values, x_values, u_values, v_values, f_values)).T

    # points_f_test = np.stack((t_ids_test, x_ids_test, t_test, x_test, u_test, v_test, h_test)).T
    # print('f', t_values.shape, x_values.shape, u_values.shape, v_values.shape, f_values.shape, points_f.shape)

    # joblib.dump({'train' : points_f, 'test' : points_f_test}, 'data/schrodinger_pts.pt')
    # # Transform data into torch loader.

    # f_dataset = DatasetClass(points_0, points_b, points_f)
    # f_dataset_test = DatasetClass(points_0, points_b, points_f_test)

    # loader = DataLoader(f_dataset, batch_size = 32, shuffle = True)
    # test_loader = DataLoader(f_dataset_test, batch_size = 32, shuffle = True)
    # for batch in loader:
    #     for k, v in batch.items():
    #         print(k, ' -> ', v.shape)

    # from tools.trainer import train

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NN(1, 3, 5, 100).to(device)
    # model.load_weights('weights/lr5e-5/basic.pth.tar')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad = True)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter = 5000)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', .2, 100)

    # train_loss, val_loss, lr_list = train(model, loader, test_loader, optimizer, scheduler, 4)

    # plt.subplot(121)
    # plt.plot(train_loss)
    # plt.subplot(122)
    # plt.plot(val_loss)
    # plt.savefig('./learning.jpg')
    # plt.show()