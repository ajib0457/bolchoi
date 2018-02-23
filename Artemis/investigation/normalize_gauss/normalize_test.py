import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz
from sympy import *
'''
#1D TEST FOR AREA----------------------
#intial values
in_val,fnl_val=-140,140
s=2

X=symbols('X')
h=(1/sqrt(2*pi*s*s))**(1)*exp(-1/(2*s*s)*(X**2))
h_int=integrate(h,(X,in_val,fnl_val))
area = h_int
print("area =", area)

kern_x=850#kernel size
X,stp=np.linspace(in_val,fnl_val,kern_x,retstep=True)
h=(1/np.sqrt(2*np.pi*s*s))**(1)*np.exp(-1/(2*s*s)*(X**2))

# Compute the area using the composite trapezoidal rule.
area = trapz(h,X)#dx is the change between each value of a.
print("area =", area)

# Compute the area using the composite Simpson's rule.
area = simps(h,dx=stp)
print("area =", area)

area=np.sum(h)*stp
print('area =', area)

#compute the area using integration of function and then applying limits and sigma

'''
#3D TEST FOR VOLUME----------------------
#initial values
in_val,fnl_val=-140,140
s=3.92

X,Y,Z=symbols('X Y Z')
h=(1/sqrt(2*pi*s*s))**(3)*exp(-1/(2*s*s)*(Y**2+X**2+Z**2))
h_int=integrate(h,(X,in_val,fnl_val),(Y,in_val,fnl_val),(Z,in_val,fnl_val))
print('volume = ',round(h_int,6))

grid_nodes=300
x,dxx=np.linspace(in_val,fnl_val,grid_nodes,retstep=True)
X,Y,Z=np.meshgrid(x,x,x)
h=(1/np.sqrt(2*np.pi*s*s))**(3)*np.exp(-1/(2*s*s)*(Y**2+X**2+Z**2))
volume=np.sum(h)*dxx**3
print('volume = ',round(volume,6))
