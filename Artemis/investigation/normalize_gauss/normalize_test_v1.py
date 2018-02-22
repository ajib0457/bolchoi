import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz


s=3.92
kern_x=850#kernel size
in_val,fnl_val=-14,14
X,stp=np.linspace(in_val,fnl_val,kern_x,retstep=True)
dxx=(1/np.sqrt(2*np.pi*s*s))**(1)*np.exp(-1/(2*s*s)*(X**2))


#a,stp=np.linspace(0,10,1000,retstep=True)
#b=a*10
# Compute the area using the composite trapezoidal rule.
area = trapz(dxx,X)#dx is the change between each value of a.
print("area =", area)

# Compute the area using the composite Simpson's rule.
area = simps(dxx,dx=stp)
print("area =", area)

area=np.sum(dxx)*stp
print('area =', area)
