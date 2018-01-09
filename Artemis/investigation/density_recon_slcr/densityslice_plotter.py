import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib import cm
import math as mth
import pandas as pd
import matplotlib.colors as mcolors
import h5py

#density field
grid_nodes=850
#f1=h5py.File("/scratch/GAMNSCM2/DTFE_1.1.1_rep/bnry_h5/output_files/DTFE_out_%d_bolchoi_rockstar_halos.a_den.h5" %grid_nodes, 'r')
#a=f1['/DTFE'][:]

f1=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/my_den/den_grid%s_halo_bin_bolchoi" %grid_nodes, 'r')
a=f1['/halo'][:]

#SLICER CODE
c=np.reshape(a,(grid_nodes,grid_nodes,grid_nodes))#density field
s=3.92
#Smooth density field
c_smoothed=ndimage.filters.gaussian_filter(c,s)

#slices options
slc=4

plt.figure(figsize=(20,20),dpi=80) 
scl_plt=30#reduce scale of density fields and eigenvalue subplots by increasing number

#Smoothed Density field
ax1=plt.subplot2grid((2,1), (0,0))    
plt.title('Smoothed Density field')
cmapp = plt.get_cmap('jet')
smth_dn_fl_plt=ax1.imshow(np.power(c_smoothed[:,slc,:],1./scl_plt),cmap=cmapp)#The colorbar will adapt to data
plt.colorbar(smth_dn_fl_plt,cmap=cmapp)
plt.xlabel('x')
plt.ylabel('y')

#Density field
ax5=plt.subplot2grid((2,1), (1,0))    
plt.title('Density field')
cmapp = plt.get_cmap('jet')
dn_fl_plt=ax5.imshow(np.power(c[:,slc,:],1./scl_plt),cmap=cmapp)#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)

plt.savefig('my_den_slices_%d_grid%d_x.png'%(slc,grid_nodes))
