# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:28:05 2020

@author: Harsha
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def post(x,t,u,save):
    
    T, X = np.meshgrid(t, x)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, T, u,cmap='jet',
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5, pad = 0.01)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$u (t,x)$', rotation=90)
    ax.zaxis.labelpad=0  
    ax.xaxis.set_ticks(np.arange(-10, 10.5, 5))
    ax.view_init(elev=27, azim=-145)
    
    if save == 0:
        pass
    else:
        plt.savefig('kdv.pdf')  
#        plt.savefig('plots/kdv.eps')  


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

dx = 0.1
cfl = 2*dx**3/(3*np.sqrt(3))

dt = 0.0003
x = np.arange(-8,8,dx)    
    
t0  =0.0
tf = 0.9
nt = int((tf-t0)/dt) + 1
t = np.linspace(t0,tf,nt)

utrue = np.zeros((len(x), nt))
mean = 0.0

si2 = 1.0e-1
si1 = np.sqrt(si2)
ne = len(x)
y = kdvEuact(x, 0, 20, 4) + kdvEuact(x, 0, 10, -4)
#y = 8.0*np.exp(-x**2)

utrue[:,0] = y

for i in range(1,nt):
    y = rk4(y, dt, dx)        
    utrue[:,i] = y 

post(x,t,utrue,1)
plt.figure()
T, X = np.meshgrid(t, x)
plt.contourf(X,T, utrue, cmap='jet')
plt.colorbar()
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$t$',fontsize=14)
plt.savefig('kdvc.pdf')

#%%

tmax = tf
nf = 100        # frequency of observation
nb = int(nt/nf)  # number of observation time

tb = np.linspace(t0,tf,nb+1)
u = np.zeros(ne)

X,T = np.meshgrid(x,t,indexing='ij')
#-----------------------------------------------------------------------------#
# generate observations
#-----------------------------------------------------------------------------#
mean = 0.0
sd2 = 1.0e-0 # added noise (variance)
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

#y = 8.0*np.exp(-x**2) + np.random.normal(mean,si1,ne)
y = ( kdvEuact(x, 0, 20, 4) + kdvEuact(x, 0, 10, -4))*1.1 + np.random.normal(mean,si1,ne)
uw[:,0] = y

for i in range(1,nt):
    y = rk4(y, dt, dx)        
    uw[:,i] = y     

#-----------------------------------------------------------------------------#
# EnKF model
#-----------------------------------------------------------------------------#    

# number of observation vector
me = 80
freq = int(ne/me)
oin = [freq*i-1 for i in range(1,me+1)]
roin = np.int32(np.linspace(0,me-1,me))

H = np.zeros((me,ne))
H[roin,oin] = 1.0

# number of ensemble 
npe = 10
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
        # compute mean of the forecast fields
        uf[:] = np.sum(ue[:,:,k],axis=1)   
        uf[:] = uf[:]/npe
        
        # compute square-root of the covariance matrix
        for n in range(npe):
            sc[:,n] = cn*(ue[:,n,k] - uf[:]) # sc ==> X'
        
        # compute DhXm data
        DhXm[:] = np.sum(ue[oin,:,k],axis=1)    
        DhXm[:] = DhXm[:]/npe
        
        # compute DhM data
        for n in range(npe):
            DhX[:,n] = cn*(ue[oin,n,k] - DhXm[:])
            
        # R = sd2*I, observation m+atrix
        cc = DhX @ DhX.T # cc ==> HPH 
        
        for i in range(me):
            cc[i,i] = cc[i,i] + sd2 # cc ==> HPH + R
        
        ph = sc @ DhX.T # ph ==> (Pf) (Dh)
                    
        ci = np.linalg.pinv(cc) # ci: inverse of cc matrix
        
        km = ph @ ci
        
        # analysis update    
        kmd = km @ (zf[:,:,kobs] - ue[oin,:,k])
        ue[:,:,k] = ue[:,:,k] + kmd[:,:]
        
        kobs = kobs+1
    
    # mean analysis for plotting
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe


fig, ax = plt.subplots(4,1,sharex=True,figsize=(6,5))

n = [0,60,100,120]
for i in range(4):
    ax[i].plot(tb,uobs[n[i],:],'ro', lw=3)
    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')
#    ax[i].plot(t,ue[n[i],0,:],'b*-')
    ax[i].grid()
    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$u_{'+str("%.1f" % x[n[i]])+'}$',fontsize=14)

ax[i].set_xlabel(r'$t$',fontsize=14)
line_labels = ['Observation','True','Erroneous','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()

plt.show() 
#fig.savefig('enkf_'+str(nf)+'_'+str(me)+'_'+str(npe)+'.pdf')

#%%
fig, ax = plt.subplots(2,1,sharex=True,figsize=(6,5))
#levels = [-1.5, 0., 1.5, 3.0, 4.5, 6., 7.5, 9.0]
T, X = np.meshgrid(t, x)
for i in range(2):
    if i ==0:
        im = ax[i].contourf(X,T, utrue,cmap='jet')
    else:    
        ax[i].contourf(X,T, ua, cmap='jet')
    ax[i].set_ylabel(r'$t$',fontsize=14)
    

ax[i].set_xlabel(r'$x$',fontsize=14)
fig.colorbar(im, ax=ax)
#fig.tight_layout()
fig.savefig('enkfc_'+str(nf)+'_'+str(me)+'_'+str(npe)+'.pdf')
plt.show()

