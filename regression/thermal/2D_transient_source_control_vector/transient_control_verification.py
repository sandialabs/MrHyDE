#!/usr/bin/python
"""
Symbolic computation
"""

# for transient single scale and mortar debugging

import sympy as sp
from sympy import pi
x = sp.Symbol('x')
y = sp.Symbol('y')
t = sp.Symbol('t')
#u = sp.sin(2*pi*x)*sp.sin(2*pi*y)*sp.sin(2*pi*t)
u = sp.sin(pi*x)*sp.sin(pi*y)*sp.sin(pi*t)
us = sp.sin(pi*x)*sp.sin(pi*y)
k = 1
f = sp.diff(u,t) + -sp.diff(k,x)*sp.diff(u,x) + -sp.diff(k,y)*sp.diff(u,y) - \
     k*(sp.diff(sp.diff(u,x),x) + sp.diff(sp.diff(u,y),y))
print("forward problem forcing = " + str(f))

adj = sp.sin(pi*x)*sp.sin(pi*y)*sp.sin(pi*t)

fadj = -sp.diff(adj,t) + -sp.diff(k,x)*sp.diff(adj,x) + -sp.diff(k,y)*sp.diff(adj,y) - \
     k*(sp.diff(sp.diff(adj,x),x) + sp.diff(sp.diff(adj,y),y))

print("adjoint problem forcing = " + str(fadj))

grad = sp.N(-sp.integrate(adj*f, (x,0,1), (y,0,1), (t,0,1)))
print("analytic gradient = " + str(grad))

obj = sp.N(sp.integrate(0.5*fadj**2, (x,0,1), (y,0,1), (t,0,1)))
print("analytic objective function = " + str(obj))
