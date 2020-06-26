# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:28:05 2020

@author: Harsha
"""

import numpy as np
import matplotlib.pyplot as plt

def kdv(u,t,dx):
    up1 = np.hstack([u[1:], u[:1]])
    up2 = np.hstack([u[2:], u[:2]])
    up3 = np.hstack([u[3:], u[:3]])
    up4 = np.hstack([u[4:], u[:4]])
    
    um1 = np.hstack([u[-1:], u[:-1]])
    um2 = np.hstack([u[-2:], u[:-2]])
    um3 = np.hstack([u[-3:], u[:-3]])
    um4 = np.hstack([u[-4:], u[:-4]])
    
    # O(h^2) Central differences
    #uu1 = (up1 - um1) / (2 * du)
    #uu3 = (up2 - 2 * up1 + 2 * um1 - um2) / (2 * du * du * du)

    # O(h^4) Central differences
    #uu1 = (-(up2 - um2) + 8 * (up1 - um1)) / (12 * du)
    #uu3 = (-(up3 - um3) + 8 * (up2 - um2) - 13 * (up1 - um1)) / (8 * du * du * du)
    
    #O(h^6) Central differences
    ux1 = ((up3 - um3) - 9 * (up2 - um2) + 45 * (up1 - um1)) / (60 * dx)
    ux3 = (7 * (up4 - um4) - 72 * (up3 - um3) + 338 * (up2 - um2) - 488 * (up1 - um1)) / (240 * dx * dx * dx)
    
    return -6 * u * ux1 - ux3


def rk4(u, dt, dx):
    k1 = dt * kdv(u, 0, dx)
    k2 = dt * kdv(u + k1 * 0.5, 0, dx)
    k3 = dt * kdv(u + k2 * 0.5, 0, dx)
    k4 = dt * kdv(u + k3, 0, dx)
    return u + 1/6. * (k1 + 2*k2 + 2*k3 + k4)

def kdvEuact(x,t,v,u0):
    a = np.cosh(0.5 * np.sqrt(v) * (x - v * t - u0))
    return v / (2 * a * a)

def kdv_exact(x, c):
    """Profile of the exact solution to the KdV for a single soliton on the real line."""
    u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)
    return u

def kdv(u, t, L):
    """Differential equations for the KdV equation, discretized in x."""
    # Compute the x derivatives using the pseudo-spectral method.
    ux = psdiff(u, period=L)
    uxxx = psdiff(u, period=L, order=3)

    # Compute du/dt.    
    dudt = -6*u*ux - uxxx

    return dudt

def kdv_solution(u0, t, L):
    """Use odeint to solve the KdV equation on a periodic domain.
    
    `u0` is initial condition, `t` is the array of time values at which
    the solution is to be computed, and `L` is the length of the periodic
    domain."""

    sol = odeint(kdv, u0, t, args=(L,), mxstep=5000)
    return sol

L = 50.0
N = 64
dx = L / (N - 1.0)
x = np.linspace(0, (1-1.0/N)*L, N)

# Set the initial conditions.
# Not exact for two solitons on a periodic domain, but close enough...


# Set the time sample grid.
T = 200  
    
t0  =0.0
tf = T
nt = 501
t = np.linspace(t0,tf,nt)
dt = t[1]-t[0]

utrue = np.zeros((len(x), nt))
mean = 0.0

si2 = 1.0e-1
si1 = np.sqrt(si2)
ne = len(x)
#y = kdvEuact(x, 0, 16, 4) + kdvEuact(x, 0, 4, -4)
y = kdv_exact(x-0.33*L, 0.75) + kdv_exact(x-0.65*L, 0.4)
#y = 8.0*np.exp(-x**2)

utrue[:,0] = y

#%%
for i in range(1,nt):
    y = rk4(y, dt, dx)        
    utrue[:,i] = y 

plt.plot(utrue[:,-1])  
plt.plot(utrue[:,0])
plt.figure()
plt.contourf(utrue)

#%%

tmax = tf
nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time

tb = np.linspace(t0,tf,nb+1)

u = np.zeros(ne)

X,T = np.meshgrid(x,t,indexing='ij')

#%%
#-----------------------------------------------------------------------------#
# generate observations
#-----------------------------------------------------------------------------#
mean = 0.0
sd2 = 1.0e-2 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

oib = [nf*k for k in range(nb+1)]

uobs = utrue[:,oib] + np.random.normal(mean,sd1,[ne,nb+1])

#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
uw = np.zeros((ne,nt))
k = 0
si2 = 1.0e-2
si1 = np.sqrt(si2)

y = 8.0*np.exp(-x**2)*1.1 + np.random.normal(mean,si1,ne)
#y =  (kdvEuact(x, 0, 16, 4) + kdvEuact(x, 0, 4, -4))*1.1 + np.random.normal(mean,si1,ne)
#y =  kdv_exact(x-0.33*L, 0.75) + kdv_exact(x-0.65*L, 0.4)*1.1 + np.random.normal(mean,si1,ne)
uw[:,0] = y

for i in range(1,nt):
    y = rk4(y, dt, dx)        
    uw[:,i] = y  


plt.figure()
plt.plot(t,utrue[5,:])
plt.plot(t,uw[5,:])
plt.plot(tb,uobs[5,:],'ko')
plt.show()

plt.figure()
plt.plot(utrue[:,-1])
plt.plot(uw[:,-1])
plt.show()    

#%%
#-----------------------------------------------------------------------------#
# EnKF model
#-----------------------------------------------------------------------------#    

# number of observation vector
me = 48
freq = int(ne/me)
oin = [freq*i-1 for i in range(1,me+1)]
roin = np.int32(np.linspace(0,me-1,me))
#print(oin)

dh = np.zeros((me,ne))
dh[roin,oin] = 1.0

H = np.zeros((me,ne))
H[roin,oin] = 1.0

# number of ensemble 
npe = 20
cn = 1.0/np.sqrt(npe-1)

z = np.zeros((me,nb+1))
zf = np.zeros((me,npe,nb+1))
DhX = np.zeros((me,npe))
DhXm = np.zeros(me)

ua = np.zeros((ne,nt)) # mean analyssi solution (to store)
uf = np.zeros(ne)        # mean forecast
sc = np.zeros((ne,npe))   # square-root of the covariance matrix
ue = np.zeros((ne,npe,nt)) # all ensambles
ph = np.zeros((ne,me))
Af = np.zeros((ne,npe))   # square-root of the covariance matrix

km = np.zeros((ne,me))
kmd = np.zeros((ne,npe))

cc = np.zeros((me,me))
ci = np.zeros((me,me))

for k in range(nb+1):
    z[:,k] = uobs[oin,k]
    for n in range(npe):
        zf[:,n,k] = z[:,k] + np.random.normal(mean,sd1,me)

# initial ensemble
k = 0
se2 = np.sqrt(sd2)
se1 = np.sqrt(se2)

for n in range(npe):
    ue[:,n,k] = uw[:,k] + np.random.normal(mean,si1,ne)       
    
ua[:,k] = np.sum(ue[:,:,k],axis=1)
ua[:,k] = ua[:,k]/npe

#%%
kobs = 1
# RK4 scheme
for k in range(1,nt):
    print(k)
    for n in range(npe):
        y = ue[:,n,k-1] 
        y = rk4(y, dt, dx)  
        un = np.copy(y)
        ue[:,n,k] = un[:] #+ np.random.normal(mean,se1,ne)
    
    if k == oib[kobs]:
        print(oib[kobs])
        # compute mean of the forecast fields
        uf[:] = np.sum(ue[:,:,k],axis=1)   
        uf[:] = uf[:]/npe
        
        # compute Af dat
        for n in range(npe):
            Af[:,n] = ue[:,n,k] - uf[:]
            
        pf = Af @ Af.T
        pf[:,:] = pf[:,:]/(npe-1)
        
        dp = dh @ pf
        cc = dp @ dh.T     

        for i in range(me):
            cc[i,i] = cc[i,i] + sd2     
        
        ph = pf @ dh.T
        
        ci = np.linalg.pinv(cc) # ci: inverse of cc matrix
        
        km = ph @ ci # compute Kalman gain
        
        # analysis update    
        kmd = km @ (z[:,kobs] - uf[oin])
        ua[:,k] = uf[:] + kmd[:]
        
        # ensemble correction
        ha = dh @ Af
        
        ue[:,:,k] = Af[:,:] - 0.5*(km @ dh @ Af) + ua[:,k].reshape(-1,1)
        
        kobs = kobs+1
    
    # mean analysis for plotting
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe

#del ue

#%%
fig, ax = plt.subplots(3,1,sharex=True,figsize=(6,5))

n = [0,14,34]
for i in range(3):
    ax[i].plot(tb,uobs[n[i],:],'ro', lw=3)
    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')
#    ax[i].plot(t,ue[n[i],0,:],'b*-')

    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig('m_'+str(me)+'.pdf')

