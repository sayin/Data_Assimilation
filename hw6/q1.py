# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:31:01 2020

@author: Harsha
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr, pinv, svd
from numpy.linalg import multi_dot, inv
from matplotlib import pyplot as plt
np.random.seed(1)


#%%
nx = 10
ny = 10
samp = 40

x = np.linspace(0,0.9,nx)
y = np.linspace(0,0.9,ny)
dx = x[1] - x[0]
dy = y[1] - y[0] 

z_loc = np.random.uniform(0.0,0.9,size=(samp,2))
H = np.zeros((samp, nx*ny))

#%%
k_loc = np.empty((0,1))
for i in range(z_loc.shape[0]):
     
     c1 = np.floor(z_loc[i,1]/dy) +  1.0     
     c2 = np.floor(z_loc[i,0]/dx) +  1.0
     print(c1,c2)
     kt = int(((c1 - 1)*nx + c2)) 
#     print(kt)
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
#w  = np.zeros((nx*ny,samp))    
#
#i = 0
#j = 0
#for k in range(nx*ny): 
#    print(i,j)           
#    w[k,:] =  np.linalg.norm(np.array([x[i],y[j]])-z_loc,axis=1)
#    if j < ny-1:        
#        j = j+1        
#    else:
#        j = 0
#        i = i+1
     
#d = 0.3
#idx = w > d
#
#w =  (d**2 - w**2)/(d**2 + w**2)
#w[idx] = 0.0
#
#s = np.sum(w,axis=1)
#w = w/np.reshape(s,[-1,1])
     
#%%        
W  = np.zeros((nx*ny,samp))  
X,Y = np.meshgrid(x,y)
X = X.reshape(-1)
Y = Y.reshape(-1)
d = 0.3
ischeme = 1

for i in range(nx*ny):
    for j in range(samp):
        r = np.sqrt((z_loc[j,0] - X[i])**2 + (z_loc[j,1] - Y[i])**2)
        if r < d:
            if ischeme == 1:
                W[i,j] = (d**2 - r**2)/(d**2 + r**2)
            elif ischeme == 2:
                W[i,j] = np.exp(-r**2/d**2)

s = np.sum(W,axis=1)
w = W/np.reshape(s,[-1,1])

        
#%%
var1 = 5.0
std1 = np.sqrt(var1)

xb = 90.0 + np.random.normal(0.0, std1, (nx*ny,))

var2 = 7.0
std2 = np.sqrt(var2)

z =  87.0 + np.random.normal(0.0, std2, (samp,))

#%%
maxit = 1000
tol   = 9e-2

xk = np.copy(xb)
for n in range(maxit):
    #xkp = np.linalg.pinv(H) @ ( H @ xk + H @ W @ (Z - H @ xk))
    xkp = xk + w @ (z - H @ xk)
    error = np.linalg.norm(xkp-xk)/(nx*ny)
    if error < tol:
        break
    print("Iter %d Error %.6f" % (n, error))
    xk = np.copy(xkp)







