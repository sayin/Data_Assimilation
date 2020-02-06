# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:56:28 2020

@author: Harsha
"""
import numpy as np
from numpy.linalg import multi_dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.linalg import lu

nx = 4
ny = 4
xl = 3
yl = 3

dx = xl/(nx-1)
dy = yl/(ny-1)

samp = 18
nvar = nx*ny

std = 0.1

v = np.random.normal(0,std,samp)

z = 70 + v

x = np.linspace(0,xl,nx)
y = np.linspace(0,yl,ny)

H = np.zeros((samp,nx*ny))
z_loc = np.zeros((samp,2))

k = 0
for j in range(nx-1):
    for i in range(ny-1):
        for p in range(2):
            zx = np.random.rand(1)
            zy = np.random.rand(1)
            z_loc[k,0] = x[j] + zx
            z_loc[k,1] = y[i] + zy
            k = k + 1   
            
#%%
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
            
     
#%%
# pseido-inverse for checking
xls_p = np.dot(np.linalg.pinv(H),z)
xls_p = xls_p.reshape(-1,1)


#%% Steepest desceny algorithm
maxit = 100000
tol = 1e-6

b = np.dot(H.T,z).reshape(-1,1)
A = np.dot(H.T,H)
xo = np.zeros((nvar,1))
xn = np.zeros((nvar,1))
r = b - np.dot(A,xo)

for k in range(maxit):
    alpha = np.dot(r.T,r)/multi_dot([r.T,A,r])
    xn = xo + alpha*r
    print(k, ' ', np.linalg.norm(xn-xo))
    if np.linalg.norm(xn-xo) < tol:
        break
    r = r - alpha*np.dot(A,r)
    xo = xn
    
# Steepest-descent gradient method solution    
xsd = xn

#%% Conjugate Gradient algorithm
b = np.dot(H.T,z).reshape(-1,1)
A = np.dot(H.T,H)
xo = np.zeros((nvar,1))
xn = np.zeros((nvar,1))
r = b - np.dot(A,xo)
p = np.copy(r)

for k in range(maxit):
    alpha = np.dot(p.T,r)/multi_dot([p.T,A,p])
    xn = xo + alpha*p
    r = r - alpha*np.dot(A,p)
    print(k, ' ', np.dot(r.T,r)[0,0])
    if np.dot(r.T,r) < tol:
        break
    beta = - multi_dot([r.T,A,p])/multi_dot([p.T,A,p])
    p = r + beta*p
    xo = xn

xcg = xn

#%%
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5))
ax.plot(xsd,'r-.',label='Steepest descent')
ax.plot(xcg,'g*-.',label='Conjugate gradient')
plt.xlim(0,15)
plt.ylim(min(xcg),max(xcg))
plt.xticks(np.arange(0, 15, step=1))
ax.legend()
plt.show()
plt.savefig('q4.pdf')