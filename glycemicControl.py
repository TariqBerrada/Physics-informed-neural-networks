import numpy as np

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
            -g : estimated initial condition of G by the network 
            -g_0 : real initial condition of G
            -x : estimated initial condition of X by the network 
            -x_0 : real initial condition of X
            -i : estimated initial condition of I by the network 
            -i_0 : real initial condition of I
    """
    return (g - g_0)**2 + (x - x_0)**2 + (i - i_0)**2

def MSE_1(G, G_point, X, T, N_f, params):
    """
        Compute the loss associated to the first equation.
        Parameters:
            -G : estimated values of G by the NN
            -G_point : estimated values of the derivative of G by the NN
            -X : estimated values of X by the NN
            -T : associated time for each value of G, G_point, ... used for the loss
            -N_f : number of points to use for the loss
            -params : parameters of the equations
    """
    loss = 0
    for i in range(N_f):
        loss += (G_point[i] + params['p1']*G[i] + X[i]*(G[i] + params['Gb']) - P(T[i]))**2
    loss /= N_f
    return loss

def MSE_2(X_point, X, I, N_f, params):
    """
        Compute the loss associated to the second equation.
        Parameters:
            -X_point : estimated values of the derivative of X by the NN
            -X : estimated values of X by the NN
            -I : estimated values of I by the NN
            -N_f : number of points to use for the loss
            -params : parameters of the equations
    """
    loss = 0
    for i in range(N_f):
        loss += (X_point[i] + params['p2']*X[i] - params['p3']*I[i])**2
    loss /= N_f
    return loss

def MSE_3(G, I_point, I, N_f, params):
    """
        Compute the loss associated to the first equation.
        Parameters:
            -G : estimated values of G by the NN
            -I_point : estimated values of the derivative of I by the NN
            -I : estimated values of I by the NN
            -N_f : number of points to use for the loss
            -params : parameters of the equations
    """
    loss = 0
    for i in range(N_f):
        loss += (I_point[i] + params['n']*(I[i] + params['Ib']) - u(G, params['strategy_number'])/(60*params['V1']))**2
    loss /= N_f
    return loss

def MSE(g, g_0, x, x_0, i, i_0, G, G_point, X, X_point, I, I_point, T, N_f):
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
            -N_f : number of points to use for the loss
    """
    params = {'p1' : 0,
              'p2' : 0.025,
              'p3' : 0.013,
              'Gb' : 4.5,
              'Ib' : 0.015,
              'n' : 5/54,
              'V1' : 12}
    MSE0 = MSE_0(g, g_0, x, x_0, i, i_0)
    MSE1 = MSE_1(G, G_point, X, T, N_f, params)
    MSE2 = MSE_2(X_point, X, I, N_f, params)
    MSE3 = MSE_3(G, I_point, I, N_f, params)
    MSE_tot = MSE0 + MSE1 + MSE2 + MSE3
    return MSE_tot