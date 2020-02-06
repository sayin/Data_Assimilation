# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:31:01 2020

@author: Harsha
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr, pinv, svd
from numpy.linalg import multi_dot, inv
from matplotlib import pyplot as plt
np.random.seed(6)


def lu_c(a_,b_):
    ## make normal eqaution (H.T*H)x = H.T*Z
    x_ = multi_dot([H.T, pinv(np.dot(H,H.T)), Zn])
    er_ = np.linalg.norm(b_-np.matmul(a_,x_))
    return x_.reshape(-1,), er_

def qr_c(a_,b_):

    q, r  = qr(a_)
    bn_ = np.matmul(q.T, b_)
    x_ = np.matmul(pinv(r),bn_)
    er_ = np.linalg.norm(b_-np.matmul(a_,x_))
    return x_.reshape(-1,), er_

def svd_c(a_,b_):

    u, s, v  = svd(a_)
    S  = np.dot(np.eye(nx*ny,samp),np.diag(1/s))
    x_ = multi_dot([v.T, S, u.T, b_])
    er_ = np.linalg.norm(b_-np.matmul(a_,x_))
    return x_.reshape(-1,), er_

#%%
nx = 4
ny = 4
samp = 4

x = np.linspace(0,0.9,4)
y = np.linspace(0,0.9,4)
dx = x[1] - x[0]
dy = y[1] - y[0] 
z_loc = np.random.rand(4,2)
H = np.zeros((samp, nx*ny))

k_loc = np.empty((0,1))
for i in range(z_loc.shape[0]):
     
     c1 = np.floor(z_loc[i,1]/dy) +  1.0     
     c2 = np.floor(z_loc[i,0]/dx) +  1.0
     print(c1,c2)
     kt = int(((c1 - 1)*nx + c2)) 
     
     aj    = z_loc[i,0] - x[int(c2)-1]
     bj    = z_loc[i,1] - y[int(c1)-1]
     ajb   = dx - aj
     bjb   = dy - bj
     
     H[i,kt-1]      = ajb*bjb
     H[i,kt+1-1]    = aj*bjb
     H[i,kt+nx-1]   = ajb*bj
     H[i,kt+nx+1-1] = aj*bj   
     
     k_loc = np.vstack((k_loc, kt)) 
     
#%% Solve Twin Experimnet
## Noise with 0 mean and variance
var = 1.0
std = np.sqrt(var)

## Noise Observations
Zn = 75 + np.random.normal(0.0, std, (samp,1))  

#%%
x_ls = np.dot(np.linalg.pinv(H),Zn)
x_lu, e_lu =  lu_c(H, Zn)
x_qr, e_qr =  qr_c(H, Zn)
x_svd, e_svd =  svd_c(H, Zn)
#%%
plt.contourf(x,y,x_ls.reshape((4,4)))
plt.scatter(z_loc[:,0],z_loc[:,1],marker='*',s=60,color='red')
plt.xlabel('$x$')
plt.ylabel(r'||$y$')
plt.title('Inv Method')
plt.savefig('q2_inv.pdf')
#%%
plt.contourf(x,y,x_ls.reshape((4,4)))
plt.scatter(z_loc[:,0],z_loc[:,1],marker='*',s=60,color='red')
plt.xlabel('$x$')
plt.ylabel(r'$y$')
plt.title('Lu Method')
plt.savefig('q2_lu.pdf')
#%%
plt.contourf(x,y,x_ls.reshape((4,4)))
plt.scatter(z_loc[:,0],z_loc[:,1],marker='*',s=60,color='red')
plt.xlabel('$x$')
plt.ylabel(r'$y$')
plt.title('QR Method')
plt.savefig('q2_qr.pdf')
#%%
plt.contourf(x,y,x_ls.reshape((4,4)))
plt.scatter(z_loc[:,0],z_loc[:,1],marker='*',s=60,color='red')
plt.xlabel('$x$')
plt.ylabel(r'$y$')
plt.title('SVD Method')
plt.savefig('q2_svd.pdf')   