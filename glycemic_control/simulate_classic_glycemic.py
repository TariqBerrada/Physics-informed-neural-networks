import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import *
import pandas as pd
def G(X_0,p_2,p_3,I_0,u,n,V_1,I_B,dt,T_f,G_0,G_d) :
    l_g=[G_0]
    l_i=[I_0]
    l_x=[X_0]
    k=0
    check=0
    u=l_g[-1]*(0.41-0.0094*l_g[-1]) 
    while k*dt < T_f :
        if (k*dt)%180==0 :
            if l_g[-1]>=6 :
                check=0
            else :
                check=1
        u=check*(1+l_g[-1]*0.22)*0.007533+(1-check)*l_g[-1]*(0.41-0.0094*l_g[-1])/60
        i_n=l_i[-1]-n*dt*(l_i[-1]+I_B)+u*dt/V_1
        x_n=l_x[-1]-dt*p_2*l_x[-1]+p_3*l_i[-1]*dt
        g_n=l_g[-1]-dt*l_x[-1]*(l_g[-1]+G_d)+0.5*dt*np.exp(-0.05*k*dt)
        l_i.append(i_n)
        l_x.append(x_n)
        l_g.append(g_n)
        k+=1

    return l_g,l_i,l_x
    
p_2=0.025
p_3=0.013
G_d=4.5
I_B=0.015
V_1=12
n=5/54
G_0=13
I_0=0.5
X_0=0
l_g,l_i,l_x=G(X_0,p_2,p_3,I_0,0,n,V_1,I_B,1,720,G_0,G_d)

df=pd.DataFrame()
df['time']=[i for i in range(721)]
df['G']=l_g
df['X']=l_x
df["I"]=l_i
plt.figure(figsize=(9,9))
sns.lineplot(x="time",y="G",data=df)
plt.show()
