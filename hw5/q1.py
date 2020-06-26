# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:56:28 2020

@author: Harsha
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(5)
import matplotlib.pyplot as plt
#
#font = {'family' : 'Times New Roman',
#        'size'   : 12}    
#plt.rc('font', **font)

#import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%    
def plot(x,un,ue):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x,un[0,:],'b',label='Initial condition')
    ax.plot(x,ue,'k',label='Exact solution')
    ax.plot(x,un[-1,:],'r-.',label='Numerical solution')
    ax.legend()
    plt.show()
    fig.tight_layout()
    
    
def plot_final(x,un,uf,uw,istd,istart):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x,uw[-1,:],'k',lw=2,label='Wrong initial condition')
    ax.plot(x,uf[-1,:],'r',lw=4,label='4DVar initial condition')
    ax.plot(x,un[-1,:],'g--',lw=2,label='True initial condition')    
    ax.grid()
    ax.legend()
    plt.show()
    fig.tight_layout()
    fig.savefig('gfinal_'+str(int(istart))+'_'+str(int(istd))+'.pdf')
    
def plot_da(x,uda0,udan,ut,z,istd,istart):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x,uda0,'k',label='Wrong initial condition')
    ax.plot(x,udan,'r',lw=4,label='4DVar initial condition')
    ax.plot(x,ut,'g--',lw=2,label='True initial condition')
#    ax.plot(x,z,'o',fillstyle='none',markersize=8,markeredgewidth=2,
#            color='b',label='Observations')
    ax.grid()
    ax.legend()
    plt.show()
    fig.tight_layout()
    fig.savefig('ginitial_'+str(int(istart))+'_'+str(int(istd))+'.pdf')
    
def plot_residual(res_history):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,5))
    ax.semilogy(res_history[:,0],res_history[:,1],'g-',label='')
    ax.set_xlabel('Itearation $(p)$')
    ax.set_ylabel('$f(x)$')
    ax.grid()
    fig.tight_layout()
    plt.show()
    fig.savefig('g_residual.pdf') 

#%%
def forward(x0,xs,k,dt,nt):
    xn = np.zeros((nt+1))
    xn[0] = x0
    for n in range(1,nt+1):
        xn[n] = xn[n-1] + dt*k*(xs - xn[n-1])
    
    return xn

def dxkm(k,dt,x):
    dxkm = np.zeros((1,1))
    dxkm[0,0] = (1-k*dt)
    return dxkm

def damj(k,xs,dt,x):
    dam = np.zeros((1,2))
    dam[0,0] = k*dt   
    dam[0,1] = (xs -x)*dt
    return dam

def dxkh():
    return np.eye(1)

#%%
istd = 1 # std = 0.01, 0.1
istart = 1 # ic = 1.2sin(x), 1.0
ischeme = 1 # scheme = Euler, RK2
iobs = 1 # observations
freq = 5

nx = 100
xl = 0
xr = 2*np.pi
dx = (xr-xl)/nx
x = np.linspace(xl, xr, nx+1)

tm = 30.0
dt = 0.1
nt = int(tm/dt)
t = np.linspace(0,tm,nt+1)

if iobs == 1:
    tobs = np.array([2.0,7.0,12.0,17.0,22.0,27.0])
    nobs = tobs.shape[0]
    ind = np.zeros(nobs, 'i')
    p = 0
    for k in range(nt+1):
        if t[k] in tobs:
            ind[p] = k
            p = p+1

if iobs == 2:
    tobs = np.array([5.0,5.5,6.0])
    nobs = tobs.shape[0]
    ind = np.zeros(tobs.shape[0], dtype='i')
    for k in range(3):
        ind[k] = np.int(np.round(tobs[k]/dt))

if iobs == 3:
    nobs = int(tm/(freq*dt))
    ind = [freq*i for i in range(1,nobs+1)]
    tobs = t[ind]
    

x0 = 1.0
xs = 11.0
kappa = 0.25

xo = forward(x0,xs,kappa,dt,nt)

mu = 0.0
if istd == 1:
    std = 0.01
elif istd == 2:
    std = 0.1
    
var = std**2

max_iter = 50
tolerance = 1e-4

res_history = np.zeros((max_iter,2))

xobs = xo[ind]
zobs = xobs + np.random.normal(mu,std,[nobs])

hk = np.zeros((nobs,3))
           
#%%
c = np.zeros((3))
c[0] = 2.0
c[1] = 10.0
c[2] = 0.3

ukl = np.zeros((nobs,1))
vkl = np.zeros((nobs,2))

for it in range(max_iter):
    xn = forward(c[0],c[1],c[2],dt,nt)    
    xobs_da = xn[ind]        
    ek = xobs_da - zobs
    
    for p in range(nobs):
        ns = ind[p]
        uk = np.eye(1)
        for i in range(ns,-1,-1):
            x = xn[i]
            uk = uk @ dxkm(c[2],dt,x)
        ukl[p,:] = uk
    
    for p in range(nobs):
        ns = ind[p]
        vk = np.zeros((1,2))
        
        for j in range(ns):
            x = xn[j]
            bj = damj(c[2],c[1],dt,x)
            
            As = np.eye(1) #(1-c[2]*dt)**(ns-j) 
            for k in range(ns,j,-1):
                x = xn[k]
                As = As @ dxkm(c[2],dt,x)
            
            vk = vk + As @ bj
        
        vkl[p,:] = vk
                
    hk = np.hstack((ukl, vkl))
    
    deltac = np.linalg.pinv(hk) @ -ek
        
    c = c + deltac
    print(it, ' ', c)
    if np.linalg.norm(deltac) < tolerance:
        break

#%%                
xda = forward(c[0],c[1],c[2],dt,nt)  
cint  = np.array([2.0,10.0,0.3])
xin = forward(cint[0],cint[1],cint[2],dt,nt)
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(6,5))
ax.plot(t, xo, 'r', lw=2,label='True')
ax.plot(t, xin, 'g--', lw=2,label='Initial')
ax.plot(t, xda, 'b--', lw=2,label='FSM')
ax.plot(tobs, xobs, 'g*', markersize=8, label='Observations')
plt.xlabel('t')
plt.ylabel('x')
ax.legend()
plt.savefig('2.pdf')
plt.show()
    
    

   