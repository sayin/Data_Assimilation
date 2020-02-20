# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:56:28 2020

@author: Harsha
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

#%% Actual states

n = 21
xe = np.zeros((n,1))
xe[0] = 0.5
for i in range(n-1):
    xe[i+1] = xe[i]**2 

#%% buliding obervation matrix
var = 0.5
std= np.sqrt(var)    
v = np.random.normal(0,std, (n-1,1))
z = xe[1:] + v
 
#%% Arbitrary states

def state_upd(x0_,lam_,obs):
    x = np.zeros((n,1))
    x[0] = x0_
    for i in range(n-1):
        x[i+1] = x[i]**2   
#        x[i+1] = 4*x[i]*(1 - x[i])
    r   = 1.0/var 
    f   = (2.0*x[1:])*r*(obs - x[1:]) 
#    f   = (4.0*(1-2*x[1:]))*r*(obs - x[1:]) 
        
    lam_[-1] = f[-1]    
    return lam_, f, x

#%% Gradient algorithm
maxit = 10000
tol = 1e-6
    
x0 = 0.8
er = np.empty((0,1))
alpha = 0.01
lam = np.zeros((n-1,1))

for k in range(maxit): 
    lam, f, x = state_upd(x0, lam, z)  
    for i in range(len(z)-2,-1,-1):        
        lam[i] =  (2*x[i+1])*lam[i+1] + f[i]
#        lam[i] =  (4.0*(1-2*x[i+1]))*lam[i+1] + f[i]
        
    cj = -(2*x[0])*lam[0]
#    cj = -(4.0*(1-2*x[0]))*lam[0]
    xn = x0 - alpha*cj    
    ertmp = np.abs(xn-x0)
    er =  np.vstack((er, ertmp))
    print(k, ' ', ertmp[0], xn[0], np.abs(0.5-x0))
    if ertmp < tol:
        break    
    x0 = xn   

#%%
sc = np.arange(0.0, 20.0, 1.0)
#fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6),sharex=True)
plt.plot(xe,color='b',linewidth=3,label='True')
plt.plot(x,'g-.',linewidth=3,label='4D VAR')
plt.plot(z,'o',fillstyle='none',markersize=6,label='Observations')
plt.legend()
plt.ylabel(r'$x^2$')
plt.show()
#plt.savefig('q2.pdf')