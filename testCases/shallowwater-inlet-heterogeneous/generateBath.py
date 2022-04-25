#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

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

dx = .1
dy = dx

x1 = np.linspace(0,Lx1,int(Lx1/dx)+1)
x2 = np.linspace(Lx1+dx,Lx1+Lx2,int(Lx2/dx))
x3 = np.linspace(Lx1+Lx2+dx,Lx1+Lx2+Lx3,int(Lx3/dx))
y1 = np.linspace(0,Ly1,int(Ly1/dy)+1)
y2 = np.linspace(Ly1+dy,Ly1+Ly2,int(Ly2/dy))
y3 = np.linspace(Ly1+Ly2+dy,Ly1+Ly2+Ly3,int(Ly3/dy))

## create a flucuating bathymetry from a real-valued
## fourier expansion

def fourierbath(xx,yy,modes,Lx,Ly):

    bath = np.zeros_like(xx)
    rng = np.random.default_rng()
    pi = np.pi

    for i in modes:
        for j in modes:
            if i==0 and j==0:
                bath += 0. ## baseline is zero (fluctuating field)
                continue
            fac = (i+j)**1.5
            a,b,c,d = rng.uniform(-3./fac,3./fac,4)

            bath += ( a*np.sin(2.*pi*xx/Lx*i)*np.sin(2.*pi*yy/Ly*j) + 
                      b*np.sin(2.*pi*xx/Lx*i)*np.cos(2.*pi*yy/Ly*j) + 
                      c*np.cos(2.*pi*xx/Lx*i)*np.sin(2.*pi*yy/Ly*j) + 
                      d*np.cos(2.*pi*xx/Lx*i)*np.cos(2.*pi*yy/Ly*j) ) 

    return bath

xx1,yy1 = np.meshgrid(x1,np.append(np.append(y1,y2),y3),indexing='ij')

## fluctuating bathymetry on top of a sloping background
bath1 = fourierbath(xx1,yy1,range(0,12),Lx1,Ly1+Ly2+Ly3)
slope = 1.5
plane = 50. - xx1*slope

bath1 = bath1 + plane

## nab the bathymetry values at the end of region 1 that
## touch region 2
ny1 = int(Ly1/dy) + 1
ny2 = int(Ly2/dy)

bath1end = bath1[ny1:ny1+ny2,-1]

bath2 = np.zeros((x2.size,y2.size))

def interpolate(bath1end,bath2,target,x):

## linearly interpolate with some tanh smoothing?

    s = .5 + .5*np.tanh( (x - x[int(x.size/2)])/.25 )

    i = 0

    for val in bath1end:
        
        bath2[:,i] = (1.-s)*val + s*target
        i += 1

    return bath2

## in region 2, we smoothly (at least in x) reach a steady value

bath2 = interpolate(bath1end,bath2,plane[-1,0],x2) ## smooth to final value

xx2,yy2 = np.meshgrid(x2,y2,indexing='ij')

xx3,yy3 = np.meshgrid(x3,np.append(np.append(y1,y2),y3),indexing='ij')

bath3 = np.zeros_like(xx3) + plane[-1,0] ## constant value

x = np.append(np.append(xx1.ravel(),xx2.ravel()),xx3.ravel())
y = np.append(np.append(yy1.ravel(),yy2.ravel()),yy3.ravel())
bath = np.append(np.append(bath1.ravel(),bath2.ravel()),bath3.ravel())

triangles = tri.Triangulation(x,y)
xt = x[triangles.triangles].mean(axis=1)
yt = y[triangles.triangles].mean(axis=1)

## need to keep from interpolating over the empty region
mask = ( ( (yt < Ly1) & ( (xt > Lx1) & (xt < Lx1 + Lx2) ) ) | 
         ( (yt > Ly1 + Ly2) & ( (xt > Lx1) & (xt < Lx1 + Lx2) ) ) )
triangles.set_mask(mask)

fig,ax = plt.subplots(subplot_kw={"projection":"3d"})
t = ax.plot_trisurf(triangles,bath)#,cmap=plt.cm.viridis,alpha=1.,antialiased=False)
ax.view_init(15,50)
ax.set_zlabel(r"Bathymetry $[L]$")
ax.set_xlabel(r"$x$ $[L]$")
ax.set_ylabel(r"$y$ $[L]$")
#fig.colorbar(t,shrink=.5)
fig.tight_layout()
plt.savefig("test.pdf")
