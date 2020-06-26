import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=1)


#%%
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
tol = 1e-3
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
plt.show()
fig.tight_layout()
fig.savefig('q21.pdf')

#%%
d = Z - H @ xb
R = sigma2o*np.eye(nobs)
xt = 90*np.ones(nvar)
xtb = xt - xb #- np.mean(xb)
B = xtb.reshape(-1,1) @ xtb.reshape(1,-1)
#B = np.cov(B)

#%%
l = np.linalg.pinv(B) + H.T @ np.linalg.pinv(R) @ H
r = H.T @ np.linalg.pinv(R) @ d
deltax = np.linalg.pinv(l) @ r

xa = xb + deltax

vmin = 80
vmax = 100
fig,ax = plt.subplots(1,1,figsize=(8,7))
cs = ax.imshow(xa.reshape(nx,ny),cmap='hot_r',vmin=vmin,vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(cs, ax=ax, shrink=0.95,ticks=np.linspace(vmin,vmax,6))
plt.show()
fig.tight_layout()
fig.savefig('q22.pdf')