import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sympy import *
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
slc=300#30 slices per 0.1 length
x,y,z=0,1,2#slice through which axis
box=np.max(halos[:,0])#subset box length
grid_den=850#density field grid resolution
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
partcls = np.delete(partcls, (0), axis=0)

#Plotting
fig, ax = plt.subplots(figsize=(15,15),dpi=100)
ax.scatter(partcls[:,0],partcls[:,2],c='r')
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')
ax.grid(True)
plt.savefig('/scratch/GAMNSCM2/bolchoi_z0/investigation/bolchoi_halosall_gd%d_slc%d_thck%sMpc.png' %(grid_den,slc,partcl_thkns))
