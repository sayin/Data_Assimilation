# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:56:28 2020

@author: Harsha
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def exact(x, t, nt):
    ue = np.zeros((nx, nt))
    for j in range(nt):
        ue[:,j] = (2.0*np.exp(t[j])*np.sin(x))/(1.0 + np.exp(2.0*t[j]) + np.cos(x)*(np.exp(2.0*t[j]) - 1.0))     
    
    return ue

def upwind(c, a, ic, nt, nx):
    u = np.zeros((nx, nt))
    u[:,0] = ic
    u[0,:] = 0.0
    u[-1,:] = 0.0
    ap = np.where(a>0,a,0)
    an = np.where(a<0,a,0) 
    
    for j in range(nt-1):       
        u[1:nx-1,j+1] = u[1:nx-1,j] - ap[1:nx-1]*(u[1:nx-1,j] - u[0:nx-2,j]) - an[1:nx-1]*(u[2:nx,j] - u[1:nx-1,j])  

    return u

#% Jacobian construcitno

def jac_up(dmk):
        
    ap = np.where(a>0,a,0)
    an = np.where(a<0,a,0)     
    
    dmk[0,0]   = 1.0 - ap[1] + an[1]
    dmk[0,1]   = -an[1]
    dmk[-1,-2] = ap[-2]
    dmk[-1,-1] = 1 - ap[-2] + an[-2]
    
    for i in range(1,len(dmk)-1):  
            dmk[i,i-1]   =  ap[i+1]
            dmk[i,i]     =  1.0 - ap[i+1] + an[i+1]
            dmk[i,i+1]   =  -an[i+1]

    return dmk

# Observation Jacobian            
def jac_obs(uk):            
    dhk = np.diag(2.0*uk)
    return dhk


def fdvar(dmk,lam,uda):
    for k, obs in enumerate(ind):
        d = jac_obs(uda[1:nx-1,obs])
        temp1 = z[:,k] - uda[:,obs]**2
        f[:,k] = np.linalg.multi_dot([d.T, np.linalg.inv(rk), temp1[1:nx-1]])
    
    lam[:,-1] = f[:,-1]
        
    for k in range(nt-2,0,-1):       
        d = jac_up(dmk)
        lam[:,k] = np.dot(d.T, lam[:,k+1]) 
        if k in ind:
#           print(k)
           lam[:,k] = np.dot(d.T, lam[:,k+1]) + f[:,int(k/10)-1]    
    
    d = jac_up(dmk)    
    grad_ = -d.T @ lam[:,1]    
    
    return grad_

def obj(uda):
    fk_ = 0.0
    for k, obs in enumerate(ind):
        temp1 = z[1:nx-1,k] - uda[1:nx-1,obs]**2
        temp2 = temp1.T @ np.linalg.inv(rk) @ temp1
        fk_ = fk_ + np.sum(temp2.reshape((-1,1)))
    return 0.5*fk_

def sol_update(uic):
    udap = upwind(c, a, uic, nt, nx)
    return udap

#%% Actual states       
    
x0 = 0.0
xl = 2.0*np.pi

nx = 101
dx = (xl - x0)/(nx-1.0)

x = np.linspace(x0, xl, nx)

t0 = 0.0
tf = 2.0

dt = 0.01
nt = int((tf - t0)/dt + 1)
t = np.linspace(t0, tf, nt)

c  = dt/dx
ic = np.sin(x)
a  = c*np.sin(x)

un = upwind(c, a, ic, nt, nx)   
ue = exact(x,t,nt)  

#plt.figure() 
#plt.plot(x, un[:,-1], label = 'True') 
#plt.plot(x, ue[:,-1], label = '4D-Var') 
#plt.legend()  
#%% Observations
mu  = 0.0
std = 0.01
nobs = 10
ind  = np.arange(10,110,10)

z = un[:,ind]**2 + np.random.normal(mu, std,(nx,nobs))

dshape = ue[1:nx-1].shape[0]
dmk = np.zeros((dshape, dshape))

rk   = std**2*np.eye((dshape))
f    = np.zeros((dshape,nobs))

lam  = np.zeros((dshape,nt))

#%% Gradeint algo

#def ic_update1(uda, grad, alpha):
##    print(alpha)
#    uic = uda[1:nx-1,0] - (alpha* grad)
#    uic = np.concatenate(([0], uic, [0]))
#
#    return uic
#
#uic0 = 1.2*np.sin(x) #np.ones((nx,))
#uda0 = upwind(c, a, uic0, nt, nx) 
##plt.plot(uda0[:,-1])
#
#max_iter = 119
#tol = 1e-2
##alpha = 1e-2
#for i in range(max_iter):
#    
#    grad0 = fdvar(dmk,lam,uda0)
#    fxk  = obj(uda0)
#    
#    alpha0 = -0.5*fxk/(grad0.T @ (-grad0))
##    alpha0 = 1.0
#    
#    uicp = ic_update1(uda0, grad0, alpha0)
#    udap = sol_update(uicp)   
#    fxpk = obj(udap)
#    
#    alpha=-((grad0.T @ -grad0)*alpha0**2)/(2.0*(fxpk - alpha0*(grad0.T @ -grad0) - fxk))    
#    
#    uicn = ic_update1(uda0, grad0, alpha)    
#    udan = sol_update(uicn) 
##    gradn = fdvar(udan)     
#    
#    er = np.linalg.norm(uicn.reshape((-1,1)) - uic0.reshape((-1,1)))
#    print(i, ' ', er, ' ',np.linalg.norm(grad0),' ', np.max(uicn))     
#    if er < tol:       
#        break
#
##    grad0 = np.copy(gradn)
#    uic0  = np.copy(uicn)     
#    uda0  = np.copy(udan) 

#%% Conjugate Gradeint
def ic_update2(uda, grad, alpha):
#    print(alpha)
    uic = uda[1:nx-1,0] + (alpha* grad)
    uic = np.concatenate(([0], uic, [0]))

    return uic

uic0 = 1.2*np.sin(x) #np.ones((nx,))
uda0 = upwind(c, a, uic0, nt, nx) 
#plt.plot(uda0[:,-1])
grad0 = fdvar(dmk,lam,uda0)
p = - grad0    
max_iter = 100
tol = 1e-2
#
for i in range(max_iter): 
    
    fxk  = obj(uda0)   
    alpha0 = -0.5*fxk/(grad0.T @ p)
#    alpha0 = 1.0
    
    uicp = ic_update2(uda0, p, alpha0)
    udap = sol_update(uicp)     
    fxpk = obj(udap)
    
    alpha=-((grad0.T @ p)*alpha0**2)/(2.0*(fxpk - alpha0*(grad0.T @ p) - fxk))      

    uicn = ic_update2(uda0, p, alpha)    
    udan = sol_update(uicn)    
    
    gradn = fdvar(dmk,lam,udan)
    
    beta = (gradn.T @ grad0)/( grad0.T @ grad0)
    
    p = -gradn + beta*p
    
    er = np.linalg.norm(uicn.reshape((-1,1)) - uic0.reshape((-1,1)))
    print(i, ' ', er, ' ', np.linalg.norm(gradn), ' ', np.max(uic0))  
    
    if er < tol:       
        break
    
    grad0 = np.copy(gradn)
    uic0  = np.copy(uicn)    
    uda0  = np.copy(udan)
    
#%%Preditions 
uda =   upwind(c, a, uicn, nt, nx)   
plt.figure()
plt.plot(x, uicn, c='g', label = 't=0 (4D Var)')  
plt.plot(x, un[:,0], c='r', label = 't=0 (True)') 
plt.scatter(x,z[:,0], label = 'Obs (t = 0.1)')
plt.legend()
#plt.savefig('q10.pdf')
#plt.savefig('q11.pdf')
#plt.savefig('q12.pdf')
#plt.savefig('q13.pdf')
#
#plt.figure()
#plt.plot(x, uda[:,-1],  c='g',  label = 't=2 (4D Var)')  
#plt.plot(x, un[:,-1], c='r',  label = 't=2 (True)') 
#plt.scatter(x,z[:,-1], label = 'Obs (t = 1)')
#plt.legend()  
#plt.savefig('q20.pdf') 
#plt.savefig('q21.pdf') 
#plt.savefig('q22.pdf')
#plt.savefig('q23.pdf')
#
#plt.figure() 
#plt.plot(x, un[:,-1], label = 'True') 
#plt.plot(x, un[:,0], label = '4D-Var') 
#plt.legend()  
#    
#%%    
#uda_da = upwind(c, a, uicn, nt, nx)
#uic_t = np.ones((nx,))
#uic_t = upwind(c, a, uic0, nt, nx) 
#
#plt.plot(uda_da[:,0])
#plt.plot(uda_da[:,-1]) 
#
#plt.plot(uic_t[:,0])    
#plt.plot(uic_t[:,-1])    
#%%
#def ic_update(uda, p, alpha):
#    uic = uda[1:nx-1,0] - (alpha* p)/np.linalg.norm(p.reshape((-1,1)))
#    uic = np.concatenate(([0], uic, [0]))
#
#    return uic
#
#
#ic = 1.2*np.sin(x)
#uda0 = upwind(c, a, ic, nt, nx) 
##
#grad0 = fdvar(uda0)
#p = -grad0
#r = -grad0
#alpha = 1.0
#
#uic0 = ic_update(uda0, p, alpha)
#udan = sol_update(uic0)
#
#max_iter = 100
#tol = 1e-5
#
#for i in range(max_iter):
#    
#    fxk  = obj(uda0)
#    fxpk = obj(udan)
#    
#    alpha = - (grad0.T @ p)/(2.0*(fxpk - grad0.T @ p - fxk))    
#    
##    uicn = ic_update1(uda0, grad0, alpha)    
##    udan = sol_update(uicn) 
##    gradn = fdvar(udan)     
##    
##    uicp = ic_update1(udan, gradn, alpha=1.0)
##    udap = sol_update(uicp)  
#    uicn = ic_update(uda0, p, alpha)    
#    udan = sol_update(uicn)    
#    
#    gradn = fdvar(udan)
#    
#    beta = (gradn.T @ grad0)/( grad0.T @ grad0)
#    
#    p = -gradn + beta*p
#    
#    er = np.linalg.norm(uicn.reshape((-1,1)) - uic0.reshape((-1,1)))
#    print(i, ' ', alpha, ' ', np.max(uic0))  
#    
##    if er < tol:       
##        break
#    
#    grad0 = np.copy(gradn)
#    uic0  = np.copy(uicn)    
#    uda0  = np.copy(udan)
    
        

    
    

   