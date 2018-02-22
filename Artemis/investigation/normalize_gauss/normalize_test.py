import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz
from mpl_toolkits.mplot3d import Axes3D


s=3.92 #standard deviation of kernal
gid_nodes=20
kern_x,kern_y,kern_z=gid_nodes,gid_nodes,gid_nodes #kernel size
in_val,fnl_val=-140,140  #value range of kernel 
#Kernel generator
X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,kern_y),np.linspace(in_val,fnl_val,kern_x),np.linspace(in_val,fnl_val,kern_z))

h=(1/np.sqrt(2*np.pi*s*s))**(3)*np.exp(-1/(2*s*s)*(Y**2+X**2+Z**2))
fig = plt.figure(figsize=(40,40))
ax = fig.add_subplot(111, projection='3d')
trans=np.where(abs(h)>0)
h[trans]=1
ax.scatter(X,Y,Z,s=h)

'''
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
'''
