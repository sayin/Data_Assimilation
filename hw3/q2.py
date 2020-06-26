# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:27:42 2020

@author: Harsha
"""

import numpy as np
np.random.seed(44)
import matplotlib.pyplot as plt

font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def plot_lorenz(q,ttrain,X,t,Z,tz,xda=[],tda=0):
    print(ttrain)
    color = ['k','red','blue']
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(8,5),sharex=True)
    axs = ax.flat
    
    for k in range(X.shape[1]):
        axs[k].plot(t,X[:,k],label=r'$x_'+str(k+1)+'$',color=color[0],
                   linewidth=2)
        axs[k].plot(tz,Z[:,k],'o',label=r'$z_'+str(k+1)+'$',color=color[1],
                    fillstyle='none',markersize=8,markeredgewidth=2)
        if xda != []:
            axs[k].plot(tda,xda[:,k],'--',label=r'$z_'+str(k+1)+'$',color=color[2],
                    fillstyle='none',markersize=8)
            axs[k].axvspan(0, ttrain, alpha=0.5, color='orange')
        axs[k].set_xlim(0,t[-1])
        axs[k].set_ylim(np.min(X[:,k])-5,np.max(X[:,k])+5)
        axs[k].set_ylabel(r'$x_'+str(k+1)+'$')
        
        #axs[k].legend()
    
    axs[k].set_xlabel(r'$t$')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    
    line_labels = ['True','Observations','4D Var']#, "ML-Train", "ML-Test"]
    plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=0.0, ncol=3, labelspacing=0.)
      
    plt.show()
    fig.tight_layout()
    fig.savefig('conj_gradientt_' +str(q)+ '.pdf', bbox_inches='tight')
    
def plot_residual(res_history):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
    ax.semilogy(res_history[:,0],res_history[:,1],'g-',label='')
    ax.set_xlabel('Itearation $(p)$')
    ax.set_ylabel('$f(x)$')
    fig.tight_layout()
    plt.show()
    fig.savefig('problem4_' +str(q)+ '.pdf') 

#%%       
def lorenz(X_init,sigma,rho,beta,dt,nt):
    X = np.zeros((nt+1,3))
    
    X[0,0] = X_init[0]
    X[0,1] = X_init[1]
    X[0,2] = X_init[2]
    
    for n in range(1,nt+1):
        x = X[n-1,0]
        y = X[n-1,1]
        z = X[n-1,2]
        X[n,0] = x - dt*sigma*(x - y)
        X[n,1] = y + dt*(rho*x - y - x*z)
        X[n,2] = z + dt*(x*y - beta*z)
    
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

def obj_function1(zk,xk,rk,nobs):
    
    obj_fun1 = 0.0
    obj_fun2 = 0.0
    for n in range(nobs):
        
        diff = zk[n,:].reshape(-1,1) - xk[n,:].reshape(-1,1)
        #obj_fun = obj_fun + np.linalg.multi_dot([diff.T,np.linalg.inv(rk),diff])    
        print(n, diff.flatten())
        obj_fun1 = obj_fun1 + diff.T @ np.linalg.inv(rk) @ diff
        obj_fun2 = obj_fun2 + diff.T @  diff
        
    return 0.5*obj_fun1, 0.5*obj_fun2

def obj_function(zk,xk,rk,nobs):

    obj_fun = 0.0
    for n in range(nobs):
        
        diff = zk[n,:].reshape(-1,1) - xk[n,:].reshape(-1,1)
        #obj_fun = obj_fun + np.linalg.multi_dot([diff.T,np.linalg.inv(rk),diff])    
        obj_fun = obj_fun + diff.T @ np.linalg.inv(rk) @ diff
        
        
    return 0.5*obj_fun

def fourdvar(zobs,xdaobs,dxk_h,rk,nobs):
    f = np.zeros((nobs,3))  
    lagr = np.zeros((nobs,3))

    for k in range(nobs):
        xk = xdaobs[k,:].reshape(-1,1)
        zk = zobs[k,:].reshape(-1,1)
        hk = zk - xk
        #fk = np.linalg.multi_dot([dxk_h.T, np.linalg.inv(rk), hk])
        fk = dxk_h.T @ np.linalg.inv(rk) @ hk
        f[k,:] = fk.reshape(1,3)
        
    lagr[-1] = f[-1,:]
    
    for k in range(nobs-2,-1,-1):
        xk = xdaobs[k,:].reshape(-1,1)
        lkp = lagr[k+1].reshape(-1,1)
        dxk_m = dxk_model(sigma,rho,beta,dt,xk)
        #lk = np.dot(dxk_m.T, lkp)  
        lk = dxk_m.T @ lkp  
        lagr[k,:] = lk.reshape(1,3) + f[k,:]
    
    x0 = xdaobs[0,:].reshape(-1,1)
    dx0_m = dxk_model(sigma,rho,beta,dt,x0)
    
    #grad = -np.dot(dx0_m.T, lagr[0,:].reshape(-1,1))
    grad = -dx0_m.T @ lagr[0,:].reshape(-1,1)
    
    return grad

def fourdvarv2(zobs,xdaobs,xda,dxk_h,rk,nobs,freq):
    f = np.zeros((nobs,3))  
    
    nt = xda.shape[0]
    lagr = np.zeros((nt,3))
    
    for k in range(nobs):
        xk = xdaobs[k,:].reshape(-1,1)
        zk = zobs[k,:].reshape(-1,1)
        hk = zk - xk
        #fk = np.linalg.multi_dot([dxk_h.T, np.linalg.inv(rk), hk])
        fk = dxk_h.T @ np.linalg.inv(rk) @ hk
        f[k,:] = fk.reshape(1,3)
        
    lagr[-1] = f[-1,:]
    
    ind = [freq*i for i in range(1,int(nttrain/freq)+1)]
    
    for k in range(nt-2,-1,-1):
        xk = xda[k,:].reshape(-1,1)
        lkp = lagr[k+1].reshape(-1,1)
        dxk_m = dxk_model(sigma,rho,beta,dt,xk)
        #lk = np.dot(dxk_m.T, lkp)  
        lk = dxk_m.T @ lkp  
        lagr[k,:] = lk.reshape(1,3)
        if k in ind:
            lagr[k,:] = lk.reshape(1,3) + f[int(k/freq)-1,:]
    
    x0 = xdaobs[0,:].reshape(-1,1)
    dx0_m = dxk_model(sigma,rho,beta,dt,x0)
    
    #grad = -np.dot(dx0_m.T, lagr[0,:].reshape(-1,1))
    grad = -dx0_m.T @ lagr[0,:].reshape(-1,1)
    
    return grad
    
#%%
sigma = 10.0
rho = 28.0
beta = 8/3

dt = 0.01
t_train = 2.0
t_max = 5.0
freq = 10

q = 0
mu = 0.0
varl = np.array([0.09])
var = varl[q]
std = np.sqrt(var)

max_iter = 100
tolerance = 1e-2

mode = 1 # 0: Static, 1 Dynamics
lr = 0.01

X_init = np.ones(3)
X_da_init = 1.1*np.ones(3)
nttrain = int(t_train/dt)
ntmax = int(t_max/dt)
nobs = int(nttrain/freq)
ttrain = np.linspace(0,t_train,nttrain+1)
ttmax = np.linspace(0,t_max,ntmax+1)

ind = [freq*i for i in range(1,int(nttrain/freq)+1)]
tobs = ttrain[ind]

rk = var*np.eye(3)

f = np.zeros((nobs,3))
lagr = np.zeros((nobs,3))

xold = X_da_init

Xtrue = lorenz(X_init,sigma,rho,beta,dt,nttrain)

xobs = Xtrue[ind]
zobs = xobs + np.random.normal(mu,std,[nobs,3])

res_history = np.zeros((max_iter,2))

dxk_h = np.eye(3,3)

#%% Conjugate gradient
    
for q in range(max_iter):
    xda = lorenz(xold,sigma,rho,beta,dt,nttrain)
    
    xdaobs = xda[ind]
    
    #gradk = fourdvar(zobs,xdaobs,dxk_h,rk,nobs)
    gradk = fourdvarv2(zobs,xdaobs,xda,dxk_h,rk,nobs,freq)
    
    if q == 0:
        pk = -np.copy(gradk)
        resk = -np.copy(gradk)
        
    ofk = obj_function(zobs,xdaobs,rk,nobs)
    
    lr = -0.5*ofk/(gradk.T @ pk)
    xkp = xdaobs + lr*pk.reshape(1,3)    
    #xkp = xdaobs + pk.reshape(1,3)
    
    ofkp = obj_function(zobs,xkp,rk,nobs)
    
    temp = np.dot(gradk.T,pk)
    lr = -temp*lr**2/(2.0*(ofkp - lr*temp - ofk))
    #lr = -temp/(2.0*(ofkp - temp - ofk))
    lr = lr[0]
    
    xnew = xold + lr*pk.flatten() #/np.linalg.norm(pk)  #np.abs(grad.flatten())
    
    xdap = lorenz(xnew,sigma,rho,beta,dt,nttrain)
    xdaobsp = xdap[ind]
    #gradkp = fourdvar(zobs,xdaobsp,dxk_h,rk,nobs)
    gradkp = fourdvarv2(zobs,xdaobsp,xdap,dxk_h,rk,nobs,freq)
    
    #beta_cg = np.dot(gradkp.T,gradk)/(np.dot(gradk.T,gradk))
    beta_pr = np.dot(gradkp.T,(gradkp - gradk))/(np.dot(gradk.T,gradk))
    beta_cg = np.max((0.0, beta_pr[0,0]))
    
    pk = -gradkp + beta_cg*pk
    
    #print(p, ' ', xold, ' ' , xnew, ' ', np.linalg.norm(grad))
    print(q, ' ', lr, ' ', np.linalg.norm(xnew-xold))
    
    if q == 0:
        res0 = np.linalg.norm(gradk)
        
    res_history[q,0] = q
    res_history[q,1] = np.linalg.norm(gradk)/res0
    
    if np.linalg.norm(xnew-xold) < tolerance:
        break

    xold = xnew

Xtrue_tmax = lorenz(X_init,sigma,rho,beta,dt,ntmax)
Xda_tmax = lorenz(xnew,sigma,rho,beta,dt,ntmax)

plot_lorenz(q,t_train,Xtrue_tmax,ttmax,zobs,tobs,Xda_tmax,ttmax)
plot_residual(res_history)