# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:55:20 2020

@author: Harsha
"""
# -*- coding: utf-8 -*-

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
nx = 101
ny = 101
samp = 20

x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
dx = x[1] - x[0]
dy = y[1] - y[0] 
#%%
z_loc = np.zeros((samp, 2))
H = np.zeros((samp, nx*ny))
var = 0.1
std = np.sqrt(var)
Zn  = np.empty((0,1))
k_loc = np.empty((0,1))

for i in range(z_loc.shape[0]):
     zx = np.random.rand(1)
     zy = np.random.rand(1)
     z = 2*zx + 4*zy + zx*zy + np.random.normal(0,std,1)
    
     z_loc[i,0] = zx
     z_loc[i,1] = zy
     c1 = np.floor(z_loc[i,1]/dy) +  1.0     
     c2 = np.floor(z_loc[i,0]/dx) +  1.0
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
     Zn = np.vstack((Zn, z))


#%%
x_lu, e_lu =  lu_c(H, Zn)
x_qr, e_qr =  qr_c(H, Zn)
x_svd, e_svd =  svd_c(H, Zn)
#%%

plt.contourf(x,y,x_lu.reshape(nx,ny))
plt.savefig('q3_lu.pdf')
plt.contourf(x,y,x_qr.reshape(nx,ny))
plt.savefig('q3_qr.pdf')
plt.contourf(x,y,x_svd.reshape(nx,ny))
plt.savefig('q3_svd.pdf')

