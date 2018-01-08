import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
import pickle
import pandas as pd
#Halos
data = pd.read_csv('/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_halos_z0',sep=r"\s+",lineterminator='\n', header = None)

data=data.as_matrix()

data=data.astype(float)

Xc=data[:,0]
Yc=data[:,1]
Zc=data[:,2]
halos=np.column_stack((Xc,Yc,Zc))

#plotting slice
        
#slices options
slc=10#30 slices per 0.1 length
x,y,z=0,1,2#slice through which axis
box=np.max(halos[:,0])#subset box length
grid_den=2000#density field grid resolution
partcl_thkns=1.3#Thickness of the particle slice, Mpc
lo_lim_partcl=1.*slc/(grid_den)*box-1.*partcl_thkns/2 #For particle distribution
hi_lim_partcl=lo_lim_partcl+partcl_thkns #For particle distributionn

#Filter particles and halos
partcls=np.array([0,0,0])

for i in range(len(halos)):
   
    if (lo_lim_partcl<halos[i,y]<hi_lim_partcl):#incremenets are in 0.00333 #wherver slc is, [x,y,z] make sure corresponds with if statement
        #density field
        result_hals=halos[i,:]
        partcls=np.row_stack((partcls,result_hals))

#add 8 points, inside PARTICLES array, on vertices of cube to avoid plotting fisheyed effect
vert_box=np.array([[box,0,0],[0,box,0],[0,0,box],[box,box,0],[box,box,box],[box,0,box],[0,box,box]])        
partcls=np.row_stack((partcls,vert_box))

#Plotting
fig=plt.figure(figsize=(40,40),dpi=150)
ax=fig.add_subplot(111, projection='3d')
#plt.title('particle Distribution')
ax.scatter(partcls[:,0],partcls[:,1],partcls[:,2],c='r')
ax.view_init(elev=0,azim=-90)#upon generating figure, usually have to rotate manually by 90 deg. clockwise 
plt.xlabel('x') 
plt.ylabel('y')
plt.savefig('snpsht_012prtcls_VELOCIhalos_gd%d_slc%d_thck%sMpc.png' %(grid_den,slc,partcl_thkns))
