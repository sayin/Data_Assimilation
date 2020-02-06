# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:32:04 2020

@author: Harsha
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr, pinv, svd
from matplotlib import pyplot as plt
np.random.seed(10)


def intgrl(gf_, l, h):
    ##Param: intergration evaluation for p*gf*exp(-p*gf) 
    #gf: freqency constants
    # l : lower limit
    # h : higher limit
    a = -(gf_*h + 1.0)*np.exp(-gf_*h)/gf_ + (gf_*l + 1)*np.exp(-gf_*l)/gf_
    return a

def lu_c(a_,b_, xe_):
    ## make normal eqaution (H.T*H)x = H.T*Z
    an_ = np.matmul(a_.T, a_)
    bn_ = np.matmul(a_.T,b_)
    lu, p  = lu_factor(an_)
    x_ = lu_solve((lu,p), bn_)
    er_ = np.linalg.norm(xe_-x_)
    return x_.reshape(-1,), er_

def qr_c(a_,b_, xe_):

    q, r  = qr(a_)
    bn_ = np.matmul(q.T, b_)
    x_ = np.matmul(pinv(r),bn_)
    er_ = np.linalg.norm(xe_ - x_)
    return x_.reshape(-1,), er_

def svd_c(a_,b_, xe_):

    u, s, v  = svd(a_)
    bn_ = np.matmul(u.T, b_)
    bn_ = np.matmul(pinv(s),bn_)
    x_ = np.matmul(v,bn_)
    er_ = np.linalg.norm(xe_ - x_)
    return x_.reshape(-1,), er_

gf = np.array([1.0/0.9, 1.0/0.7, 1.0/0.5, 1.0/0.3, 1.0/0.2])

a11 = intgrl(gf,0.5, 1.0)  ##lim 0.5, 1.0
a12 = intgrl(gf,0.2, 0.5)  ##lim 0.2, 0.5
a13 = intgrl(gf,0.0, 0.2)  ##lim 0.0, 0.2

## Assembly H matrix
H = np.array([a11,a12,a13]).T

#%%
## Solve Forward Problem

## Assume T values
Tx = np.array([0.9, 0.85, 0.875]).reshape(-1,1)

## Calculate Obersrvation Z
Z = np.matmul(H,Tx)

#%%
# Solve Twin Experimnet
## Noise with 0 mean and variance
var = np.array([0.0, 0.1, 0.4,  0.8, 1.0, 1.2])
std = np.sqrt(var)

## Noise Observations
#Zn = Z + np.random.normal(0.0, std) 

#%%

T_lu = np.empty((0,Tx.shape[0]))
er_lu = np.empty((0,1))

for i in range(std.shape[0]):
    Zn = Z +  np.random.normal(0, std[i])
    xtmp, ertmp =  lu_c(H, Zn, Tx)
    T_lu = np.vstack((T_lu, xtmp))
    er_lu =  np.vstack((er_lu, ertmp))
    
plt.plot(var, er_lu)
plt.xlabel('$\sigma^2$')
plt.ylabel(r'||$\Omega||^2$')
plt.title('LU Method')
plt.savefig('q1_lu.pdf')
#%%    
T_qr = np.empty((0,Tx.shape[0]))
er_qr = np.empty((0,1))
    
for i in range(std.shape[0]):
    Zn = Z +  np.random.normal(0, std[i])
    xtmp, ertmp =  qr_c(H, Zn, Tx)
    T_qr = np.vstack((T_qr, xtmp))
    er_qr =  np.vstack((er_qr, ertmp))
    
plt.plot(var, er_qr)
plt.xlabel('$\sigma^2$')
plt.ylabel(r'||$\Omega||^2$')
plt.title('QR Method')
plt.savefig('q1_qr.pdf')
#%%
T_svd = np.empty((0,Tx.shape[0]))
er_svd = np.empty((0,1))
    
for i in range(std.shape[0]):
    Zn = Z +  np.random.normal(0, std[i])
    xtmp, ertmp =  qr_c(H, Zn, Tx)
    T_svd = np.vstack((T_svd, xtmp))
    er_svd =  np.vstack((er_svd, ertmp))
    
plt.plot(var, er_svd)
plt.xlabel('$\sigma^2$')
plt.ylabel(r'||$\Omega||^2$')
plt.title('SVD Method')
plt.savefig('q1_svd.pdf')    
    