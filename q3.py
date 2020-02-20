# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:56:28 2020

@author: Harsha
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

#%% Actual states       
    
def lorenz(X_init,sigma,rho,beta,dt,nt):
    X = np.zeros((nt+1,3))
    
    X[0,0] = X_init[0]
    X[0,1] = X_init[1]
    X[0,2] = X_init[2]
    
    for n in range(1,nt+1):
        X[n,0] = X[n-1,0] - dt*sigma*(X[n-1,0] - X[n-1,1])
        X[n,1] = X[n-1,1] + dt*(rho*X[n-1,0] - X[n-1,1] - X[n-1,0]*X[n-1,2])
        X[n,2] = X[n-1,2] + dt*(X[n-1,0]*X[n-1,1] - beta*X[n-1,2])
    
    return X

def dxk_model(sigma,rho,beta,dt,xk):
    dxk_m = np.zeros((3,3))
    dxk_m[0,0] = 1 - sigma*dt
    dxk_m[0,1] = sigma*dt
    dxk_m[0,2] = 0
    dxk_m[1,0] = rho*dt - dt*xk[2]
    dxk_m[1,1] = 1 - dt
    dxk_m[1,2] = -dt*xk[0]
    dxk_m[2,0] = dt*xk[1]
    dxk_m[2,1] = dt*xk[0]
    dxk_m[2,2] = 1-beta*dt
    
    return dxk_m

#%%
n = 41
X_init = np.array([1.0, 1.0, 1.0])

sigma = 10.0
rho = 28.0
beta = 8/3

dt = 0.01
t_train = 0.4
t_max = 5.0

ntmax = int(t_max/dt)
nttrain = int(t_train/dt)


Xtrue = lorenz(X_init,sigma,rho,beta,dt,nttrain)

#%%
freq = 2
nobs = int(nttrain/freq)
ttrain = np.linspace(0,t_train,nttrain+1)
ttmax = np.linspace(0,t_max,ntmax+1)


ind = [freq*i for i in range(1,int(nttrain/freq)+1)]
tobs = ttrain[ind]

x_da_init = np.empty((0,3))
#%%
mu = 0.0
var = 0.04
std = np.sqrt(var)
xobs = Xtrue[ind]
zobs = xobs + np.random.normal(mu,std,[nobs,3])   

rk = var*np.eye(3)
f    = np.zeros((nobs,3))
lagr = np.zeros((nobs,3))

max_iter = 10000
tolerance = 1e-5
lr = 0.0001

X_da_init = 1.1*np.ones(3)
xold = X_da_init   

grado = 0.0
for p in range(max_iter):
    Xda = lorenz(xold,sigma,rho,beta,dt,nttrain)    
    xdaobs = Xda[ind]
    
    for k in range(nobs):
        xk = xdaobs[k,:].reshape(-1,1)
        dxk_m = dxk_model(sigma,rho,beta,dt,xk)
        hk = zobs[k,:].reshape(-1,1) - xk
        fk = np.linalg.multi_dot([dxk_m.T, np.linalg.inv(rk), hk])
        f[k,:] = fk.flatten()
        
    lagr[-1] = f[-1,:]
    
    for k in range(nobs-2,-1,-1):
        xk = xdaobs[k,:].reshape(-1,1)
        lkp = lagr[k+1].reshape(-1,1)
        dxk_m = dxk_model(sigma,rho,beta,dt,xk)
        lk = np.dot(dxk_m.T, lkp)  
        lagr[k,:] = lk.flatten() + f[k,:]
    
    x0 = xdaobs[0,:].reshape(-1,1)
    dx0_m = dxk_model(sigma,rho,beta,dt,x0)
    
    gradn = -np.dot(dx0_m.T, lagr[0,:].reshape(-1,1))
    
    xnew = xold - lr*gradn.flatten()/np.linalg.norm(gradn)  #np.abs(grad.flatten())
    
    #print(p, ' ', xold, ' ' , xnew, ' ', np.linalg.norm(grad))
    print(p, ' ', np.linalg.norm(xnew-xold), xold[0], xold[1], xold[2])
    
    if np.linalg.norm(xnew-xold) < tolerance:
        break
    
    grado = gradn
    xold = xnew


x_da_init = np.vstack((x_da_init, xnew))

Xtrue_tmax = lorenz(X_init,sigma,rho,beta,dt,ntmax)
Xda_tmax   = lorenz(xnew,sigma,rho,beta,dt,ntmax)

#%%
def plot_lorenz(ttrain,X,t,Z,tz,xda=[],tda=0):
#    print(ttrain)
    color = ['red','green','blue']
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,6),sharex=True)
    axs = ax.flat
    
    for k in range(X.shape[1]):
        axs[k].plot(t,X[:,k],label=r'$x_'+str(k+1)+'$',color=color[0],
                   linewidth=2)
        axs[k].plot(tz,Z[:,k],'o',label=r'$z_'+str(k+1)+'$',color=color[1],
                    fillstyle='none',markersize=7,markeredgewidth=2)
        if xda != []:
            axs[k].plot(tda,xda[:,k],'--',label=r'$z_'+str(k+1)+'$',color=color[2],
                    fillstyle='none',markersize=8)
            axs[k].axvspan(0, ttrain, alpha=0.5, facecolor='0.5')
        axs[k].set_xlim(0,t[-1])
        axs[k].set_ylim(np.min(X[:,k])-5,np.max(X[:,k])+5)
        if k == 0:
            axs[k].set_ylabel(r'$x$')
        elif k == 1:
            axs[k].set_ylabel(r'$y$')        
        elif k == 2:
            axs[k].set_ylabel(r'$z$')        #axs[k].legend()
   
    axs[k].set_xlabel(r'$t$ (s)')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.125)
    
    line_labels = ['True','Observations','4D Var']#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=4, labelspacing=0.)
#    plt.savefig('q3_s2.pdf')  
    plt.show()
    
plot_lorenz(t_train,Xtrue_tmax,ttmax,zobs,tobs,Xda_tmax,ttmax)
  
