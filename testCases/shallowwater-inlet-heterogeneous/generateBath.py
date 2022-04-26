#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import UnivariateSpline as US

## we will generate an idealized coastal inlet similar to Samii 2019

#   L_x1          L_x3
#############    #####
#           #L_x2#   # L_y1
#           ######   #
#                    # L_y2
#           ######   #
#           #    #   # L_y3
#############    #####

# In the first region we will have a heterogeneous bathymetry
# in the second the bathymetry will level off to a homogeneous value 
# in the third, it will stay that homogeneous value

# TODO made up units for now...
Lx1 = 10.
Lx2 = 2.
Lx3 = 3.
Ly1  = 6.
Ly2  = 3.
Ly3  = 6.

dx = .05
dy = dx

x1 = np.linspace(0,Lx1,int(Lx1/dx)+1)
x2 = np.linspace(Lx1+dx,Lx1+Lx2,int(Lx2/dx))
x3 = np.linspace(Lx1+Lx2+dx,Lx1+Lx2+Lx3,int(Lx3/dx))
y1 = np.linspace(0,Ly1,int(Ly1/dy)+1)
y2 = np.linspace(Ly1+dy,Ly1+Ly2,int(Ly2/dy))
y3 = np.linspace(Ly1+Ly2+dy,Ly1+Ly2+Ly3,int(Ly3/dy))

ny1 = int(Ly1/dy) + 1
ny2 = int(Ly2/dy)

## create a fluctuating bathymetry from a real-valued
## fourier expansion

def fourierbath(xx,yy,modes,Lx,Ly):

    bath = np.zeros_like(xx)
    dbath = np.zeros( xx.shape + (2,) )
    rng = np.random.default_rng()
    pi = np.pi

    for i in modes:
        for j in modes:
            if i==0 and j==0:
                bath += 0. ## baseline is zero (fluctuating field)
                dbath += 0. 
                continue
            fac = (i+j)**1.5
            a,b,c,d = rng.uniform(-3./fac,3./fac,4)

            bath += ( a*np.sin(2.*pi*xx/Lx*i)*np.sin(2.*pi*yy/Ly*j) + 
                      b*np.sin(2.*pi*xx/Lx*i)*np.cos(2.*pi*yy/Ly*j) + 
                      c*np.cos(2.*pi*xx/Lx*i)*np.sin(2.*pi*yy/Ly*j) + 
                      d*np.cos(2.*pi*xx/Lx*i)*np.cos(2.*pi*yy/Ly*j) ) 

            facx = 2.*pi*i/Lx
            facy = 2.*pi*j/Ly

            ## d/dx

            dbath[:,:,0] += ( a*np.cos(2.*pi*xx/Lx*i)*np.sin(2.*pi*yy/Ly*j) + 
                              b*np.cos(2.*pi*xx/Lx*i)*np.cos(2.*pi*yy/Ly*j) - 
                              c*np.sin(2.*pi*xx/Lx*i)*np.sin(2.*pi*yy/Ly*j) - 
                              d*np.sin(2.*pi*xx/Lx*i)*np.cos(2.*pi*yy/Ly*j) ) * facx
            ## d/dy

            dbath[:,:,1] += ( a*np.sin(2.*pi*xx/Lx*i)*np.cos(2.*pi*yy/Ly*j) - 
                              b*np.sin(2.*pi*xx/Lx*i)*np.sin(2.*pi*yy/Ly*j) + 
                              c*np.cos(2.*pi*xx/Lx*i)*np.cos(2.*pi*yy/Ly*j) - 
                              d*np.cos(2.*pi*xx/Lx*i)*np.sin(2.*pi*yy/Ly*j) ) * facy

    return bath,dbath

## create a slab of size [ Lx1 + Lx2, Ly1 + Ly2 + Ly3 ]
## later we'll remove the cutout portions
xx1,yy1 = np.meshgrid(x1,np.append(np.append(y1,y2),y3),indexing='ij')
xx12,yy12 = np.meshgrid(np.append(x1,x2),np.append(np.append(y1,y2),y3),indexing='ij')

## fluctuating bathymetry on top of a sloping background
bath12,dbath12 = fourierbath(xx12,yy12,range(0,12),Lx1+Lx2,Ly1+Ly2+Ly3)
slope = 1.5
plane = 50. - xx12*slope

bath12 = bath12 + plane
dbath12 = dbath12
dbath12[:,:,0] -= slope


def interpolate(bath12,dbath12,target,x,x0):

    bath12smooth = np.zeros_like(bath12)
    dbath12smooth = np.zeros( bath12.shape+(2,) )

## linearly interpolate with some tanh smoothing?

    delta = .25

    s = .5 + .5*np.tanh( (x - x0)/delta )
    ds = .5/(delta*np.cosh( (x - x0)/delta )**2)

    for j in range(bath12.shape[1]):
        
        bath12smooth[:,j] = (1.-s)*bath12[:,j] + s*target
        dbath12smooth[:,j,0] = (1.-s)*dbath12[:,j,0] - bath12[:,j]*ds + ds*target
        dbath12smooth[:,j,1] = (1.-s)*dbath12[:,j,1]

    return bath12smooth,dbath12smooth

## in region 2, we smoothly (at least in x) reach a steady value
## to do so, we grab values from region 1 too

bath12smooth,dbath12smooth = interpolate(bath12,dbath12,plane[-1,0],np.append(x1,x2),x0=Lx1+Lx2/2.) ## smooth to final value

## fill in individual regions (cutout the inlet)

bath1 = bath12smooth[0:x1.size,:]
dbath1 = dbath12smooth[0:x1.size,:,:]

bath2 = bath12smooth[x1.size:,ny1:ny1+ny2]
dbath2 = dbath12smooth[x1.size:,ny1:ny1+ny2,:]

xx2,yy2 = np.meshgrid(x2,y2,indexing='ij')

xx3,yy3 = np.meshgrid(x3,np.append(np.append(y1,y2),y3),indexing='ij')

bath3 = np.zeros_like(xx3) + plane[-1,0] ## constant value
dbath3 = np.zeros( bath3.shape + (2,) )

x = np.append(np.append(xx1.ravel(),xx2.ravel()),xx3.ravel())
y = np.append(np.append(yy1.ravel(),yy2.ravel()),yy3.ravel())
bath = np.append(np.append(bath1.ravel(),bath2.ravel()),bath3.ravel())
dbath = np.zeros( bath.shape + (2,) )
dbath[:,0] = np.append(np.append(dbath1[:,:,0].ravel(),dbath2[:,:,0].ravel()),dbath3[:,:,0].ravel())
dbath[:,1] = np.append(np.append(dbath1[:,:,1].ravel(),dbath2[:,:,1].ravel()),dbath3[:,:,1].ravel())

triangles = tri.Triangulation(x,y)
xt = x[triangles.triangles].mean(axis=1)
yt = y[triangles.triangles].mean(axis=1)

## need to keep from interpolating over the empty region
mask = ( ( (yt < Ly1) & ( (xt > Lx1) & (xt < Lx1 + Lx2) ) ) | 
         ( (yt > Ly1 + Ly2) & ( (xt > Lx1) & (xt < Lx1 + Lx2) ) ) )
triangles.set_mask(mask)

fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
t = ax.plot_trisurf(triangles,bath)#,cmap=plt.cm.viridis,alpha=1.,antialiased=False)
ax.view_init(30,75)
ax.set_zlabel(r"Bathymetry $[L]$")
ax.set_xlabel(r"$x$ $[L]$")
ax.set_ylabel(r"$y$ $[L]$")
#fig.colorbar(t,shrink=.5)
fig.tight_layout()
plt.savefig("test.pdf")

#plt.clf()
#
##inds = np.where(y==7.5) 
#for val in [Lx1/2.,Lx1,Lx1+Lx2/2.,Lx1+Lx2]:
##for val in [Ly1+Ly2/4.,Ly1+Ly2/2.]:
#    inds = np.where(x==val)
#    #inds = np.where(y==val)
#    coords = y[inds]
#    #coords = x[inds]
#    #plt.plot(coords,dbath[:,0][inds])
#    plt.plot(coords,dbath[:,1][inds])
#    #plt.plot(coords,bath[inds])
#
#    sp = US(coords,bath[inds],k=5,s=0)
#    plt.plot(coords,sp.derivative()(coords),'--')
#    #plt.plot(coords,sp(coords),'--')
#
#plt.tight_layout()
#plt.savefig('test.pdf')
