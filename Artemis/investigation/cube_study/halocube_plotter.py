import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
import pickle
import h5py

grid_nodes=850#density field grid resolution
box_sz_x_min=50
box_sz_x_max=200
box_sz_y_min=200
box_sz_y_max=250
box_sz_z_min=50
box_sz_z_max=100
#f=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/dtfe_box/halo_box_cutout_sz%s_grid%s.h5" %(box_sz,grid_nodes), 'r')
f=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/halo_box_cutout_sz%s_%s_%s_%s_%s_%s_grid%s.h5" %(box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max,grid_nodes), 'r')
data=f['/group/x'][:]
f.close()

Xc=data[:,0]
Yc=data[:,1]
Zc=data[:,2]
hals=np.column_stack((Xc,Yc,Zc))


#add 8 points, inside PARTICLES array, on vertices of cube to avoid plotting fisheyed effect
#vert_box=np.array([[box,0,0],[0,box,0],[0,0,box],[box,box,0],[box,box,box],[box,0,box],[0,box,box]])        
#partcls=np.row_stack((partcls,vert_box))

#FILTER OUT PARTICLES
#mask_x=np.where()
#mask_y=np.where()
#mask_z=np.where()
#Plotting
fig=plt.figure(figsize=(20,20),dpi=100)
ax=fig.add_subplot(111, projection='3d')
#plt.title('particle Distribution')
ax.scatter(hals[:,0],hals[:,1],hals[:,2],c='r',s=0.1)
#ax.view_init(elev=0,azim=-90)#upon generating figure, usually have to rotate manually by 90 deg. clockwise 
plt.xlabel('x') 
plt.ylabel('y')
plt.savefig('snpsht_012prtcls_VELOCIhalos_g_slc_thcc_xyflip.png' )
