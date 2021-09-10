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