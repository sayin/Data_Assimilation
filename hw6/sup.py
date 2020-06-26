# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:04:14 2020

@author: Harsha
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr, pinv, svd
from numpy.linalg import multi_dot, inv
from matplotlib import pyplot as plt
np.random.seed(6)

nx = 10
ny = 10
xl = 9
yl = 9
xlo = 9
ylo = 9

dx = xl/(nx-1)
dy = yl/(ny-1)

nobs = 40
nvar = nx*ny

mu = 0
sigma2 = 5.0
std = np.sqrt(sigma2)

sigma2o = 7.0
stdo = np.sqrt(sigma2o)

maxiter = 50
tol = 3e-2
ischeme = 2

#%%
v = np.random.normal(mu,std,nvar)
xb = 90 + v

v = np.random.normal(mu,stdo,nobs)
Z = 87 + v

x = np.linspace(0,xl,nx)
y = np.linspace(0,yl,ny)

H = np.zeros((nobs,nvar))
Zx = np.zeros((nobs))
Zy = np.zeros((nobs))

Zx = np.random.rand(nobs)*xlo
Zy = np.random.rand(nobs)*ylo

#%%
for n in range(nobs):
    zx = np.random.rand(1)*xl
    zy = np.random.rand(1)*yl
    
    Zx[n] = zx
    Zy[n] = zy
    
    
    j = int(zx/dx) 
    i = int(zy/dy)

    a = (zx - x[j])/dx
    abar = (x[j+1] - zx)/dx
    b = (zy - y[i])/dy
    bbar = (y[i+1] - zy)/dy

    k = i*nx + j
    
    H[n,k] = abar*bbar
    H[n,k+1] = a*bbar
    H[n,k+nx] = abar*b
    H[n,k+nx+1] = a*b

#%%
fig,ax = plt.subplots(1,1,figsize=(8,7))
ax.plot(Zx,Zy,'go',fillstyle='none',markersize=6,markeredgewidth=3)
ax.grid()
ax.set_aspect('equal')
ax.set_xlim([0,9])
ax.set_ylim([0,9])
ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax.set_yticks([0,1,2,3,4,5,6,7,8,9])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
fig.tight_layout()
fig.savefig('q11.pdf')

#%% Cressman weighing scheme
d = 3
W = np.zeros((nvar,nobs))

X,Y = np.meshgrid(x,y)
X = X.reshape(-1)
Y = Y.reshape(-1)

#%%
for i in range(nvar):
    for j in range(nobs):
        r = np.sqrt((Zx[j] - X[i])**2 + (Zy[j] - Y[i])**2)
        if r < d:
            if ischeme == 1:
                W[i,j] = (d**2 - r**2)/(d**2 + r**2)
            elif ischeme == 2:
                W[i,j] = np.exp(-r**2/d**2)
                #d = 0.99*d

#aa = x.reshape(-1,1) - Zx.reshape(1,-1)
#bb = y.reshape(-1,1) - Zy.reshape(1,-1)
#rr = np.sqrt(aa**2 + bb**2)
#if ischeme == 1:
#    W = (d**2 - rr**2)/(d**2 + rr**2)
#elif ischeme == 2:
#    W = np.exp(-rr**2/d**2)
    
#W = np.where(rr <= d, W, 0)
S = np.sum(W,axis=1)
W = W/np.reshape(S,[-1,1])

sp = np.eye(nobs) - H @ W
eigs = np.linalg.eigvals(sp)
sprad =  np.max(np.abs(eigs))
print('Spectral radius = ', sprad)

#%%
S = np.sum(W,axis=1)
W = W/S.reshape(-1,1)       

#%%
T = H @ W
y = np.linalg.pinv(T) @ (Z - H @ xb)
xa = xb + W @ y

fig,ax = plt.subplots(1,1,figsize=(8,7))
ax.plot(np.real(eigs),'bo-',fillstyle='none',markersize=12,markeredgewidth=3)
ax.set_ylabel(r'$\lambda$')
plt.show()
fig.tight_layout()
#fig.savefig('p2_egen_b.pdf')

#%%
xk = np.copy(xb)
for n in range(maxiter):
    #xkp = np.linalg.pinv(H) @ ( H @ xk + H @ W @ (Z - H @ xk))
    xkp = xk + W @ (Z - H @ xk)
    error = np.linalg.norm(xkp-xk)/nvar
    if error < tol:
        break
    print("Iter %d Error %.6f" % (n, error))
    xk = np.copy(xkp)

#%%
vmin = 80
vmax = 100
fig,ax = plt.subplots(1,1,figsize=(8,7))
cs = ax.imshow(xk.reshape(nx,ny),cmap='hot',vmin=vmin,vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(cs, ax=ax, shrink=0.95,ticks=np.linspace(vmin,vmax,6))
plt.show()
fig.tight_layout()
fig.savefig('q121.pdf')