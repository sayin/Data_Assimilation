# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:56:28 2020

@author: Harsha
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

#%% Actual states

n = 51
xe = np.zeros((n,1))
xe[0] = 1.0
a = 1.0
for i in range(n-1):
    xe[i+1] = a*xe[i] 

#%% buliding obervation matrix
var = 0.5
std= np.sqrt(var)    
v = np.random.normal(0,std, (n-1,1))
z = xe[1:] + v

o1 = np.copy(z)
o2 = np.vstack((z[0],z[4:n-1:5]))
o3 = np.vstack((z[0],z[9:n-1:10]))


#%% Arbitrary states

def f_calc(obs, x_, level):
    r   = 1/var 
    if level ==1:
        f   = r*(obs - x_[1:]) 
    elif level == 2:
        x_ =  np.vstack((x_[1],x_[5:n:5]))
        f =   r*(obs - x_)
    elif level == 3:
        x_ =  np.vstack((x_[1],x_[10:n:10]))
        f =   r*(obs - x_)   
        
    return f

def state_upd(x0_,lam_, obs, level):
    x = np.zeros((n,1))
    x[0] = x0_
    a    = 1.0
    for i in range(n-1):
        x[i+1] = a*x[i]        
    
    f   = f_calc(obs, x, level)
        
    lam_[-1] = f[-1]    
    return lam_, f, x

#%% Gradient algorithm
maxit = 100000
tol = 1e-6
    
x0 = 0.5
er = np.empty((0,1))
alpha = 0.001
lam = np.zeros((n-1,1))

n_obs = 3
obs = o3
for k in range(maxit):    
    for i in range(len(obs)-2,-1,-1):
        lam, f, x = state_upd(x0, lam, obs, n_obs)
        lam[i] =  lam[i+1] + f[i]
        
    cj = -lam[0]
    xn = x0 - alpha*cj    
    ertmp = np.abs(xn-x0)
    er =  np.vstack((er, ertmp))
    print(k, ' ', ertmp[0], xn[0], np.abs(1.0-x0))
    if ertmp < tol:
        break    
    x0 = xn   

#%%
sc = np.arange(0.0,len(obs), 1.0)
#fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6),sharex=True)
plt.plot(xe,color='r',linewidth=3,label='True')
plt.plot(x,'b-.',linewidth=3,label='4D VAR')
plt.plot(obs,'o',fillstyle='none',markersize=6,label='Obsv'+'('+str(len(obs))+')')
plt.legend()
plt.ylabel(r'$x$')
plt.show()
#plt.savefig('q1_'+ str(len(obs))+'.pdf')
