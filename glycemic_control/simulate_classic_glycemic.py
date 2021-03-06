import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from math import *
import pandas as pd

## Here we use a classical method to compute the solution of the hybrid system

def G(X_0,p_2,p_3,I_0,u,n,V_1,I_B,dt,T_f,G_0,G_d) :
    l_g=[G_0]
    l_i=[I_0]
    l_x=[X_0]
    k=0
    check=0
    ## Initial condition for u
    u=l_g[-1]*(0.41-0.0094*l_g[-1]) 
    while k*dt < T_f :
        ## In the 3rd strategy, we check the state of the system every 3 hours
        if (k*dt)%180==0 :
            ## We check G to see if we change the formula for u 
            if l_g[-1]>=6 :
                check=0
            else :
                check=1
        ## There was a mistake in the website, for the condition G>6 mmol, it was expressed in U.min^-1,
        ## however, after testing, it was in U.h^-1, so we divided it by 60     
        u=check*(1+l_g[-1]*0.22)*0.007533+(1-check)*l_g[-1]*(0.41-0.0094*l_g[-1])/60
        ## We compute the next step of the system
        i_n=l_i[-1]-n*dt*(l_i[-1]+I_B)+u*dt/V_1
        x_n=l_x[-1]-dt*p_2*l_x[-1]+p_3*l_i[-1]*dt
        g_n=l_g[-1]-dt*l_x[-1]*(l_g[-1]+G_d)+0.5*dt*np.exp(-0.05*k*dt)
        l_i.append(i_n)
        l_x.append(x_n)
        l_g.append(g_n)
        k+=1

    return l_g,l_i,l_x
## parameter values
p_2=0.025
p_3=0.013
G_d=4.5
I_B=0.015
V_1=12
n=5/54

## Initial conditions
G_0=13
I_0=0.5
X_0=0

## Compute the solution of the system, with T_max=720 min, dt=1 min, as in the website
l_g,l_i,l_x=G(X_0,p_2,p_3,I_0,0,n,V_1,I_B,1,720,G_0,G_d)

df=pd.DataFrame()
df['time']=[i for i in range(721)]
df['G']=l_g
df['X']=l_x
df["I"]=l_i
# plt.figure(figsize=(9,9))
# sns.lineplot(x="time",y="G",data=df)
# plt.show()
plt.figure()
plt.plot(df['time'][:180], df['G'][:180], 'b')
plt.plot(df['time'][180:], df['G'][180:], 'r')

plt.scatter(df['time'][180], df['G'][180])

print(df['G'][175])
print('100', df['G'][100])
print('50', df['G'][50])
print('500', df['G'][500])

plt.legend(['state 1 : $u = u_1$', 'state 2 : $u = u2$', 'transition'])
plt.xlabel('time $(min)$')
plt.ylabel('G $(mmol.L^{-1})$')
plt.savefig('grph.png')
plt.show()
