import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import pandas as pd
f=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/halo_color_500part.h5", 'r')
data_in=f['/group/x'][:]#already filtered for >=500
mass = pd.read_csv('/import/oth3/ajib0457/bolchoi_z0/catalogs/rockstar/bolchoi_DTFE_rockstar_halos_z0_xyz_m_j',sep=r"\s+",lineterminator='\n', header = None)
mass=mass.as_matrix()

rad = pd.read_csv('/import/oth3/ajib0457/bolchoi_z0/catalogs/rockstar/bolchoi_DTFE_rockstar_halos500_m_r',sep=r"\s+",lineterminator='\n', header = None)
rad=rad.as_matrix()#already filtered for >=500

partcl_500=np.where((mass[:,3]/(1.35*10**8))>=500)#filter out halos with <500 particles
mass=mass[partcl_500]
data=np.column_stack((data_in,mass[:,3],rad[:,1]))
#data=np.column_stack((data,rad))
#slices options
slc=10#30 slices per 0.1 length
x,y,z=0,1,2#slice through which axis
box=np.max(data[:,0])#subset box length
grid_den=2000#density field grid resolution
partcl_thkns=0.125#Thickness of the particle slice, Mpc
lo_lim_partcl=1.*slc/(grid_den)*box-1.*partcl_thkns/2 #For particle distribution
hi_lim_partcl=lo_lim_partcl+partcl_thkns #For particle distributionn

#Filter particles and halos
partcls=np.array([0,0,0,0,0,0])

for i in range(len(data)):
   
    if (lo_lim_partcl<data[i,y]<hi_lim_partcl):#incremenets are in 0.00333 #wherver slc is, [x,y,z] make sure corresponds with if statement
        #density field
        result_hals=data[i,:]
        partcls=np.row_stack((partcls,result_hals))

#add 8 points, inside PARTICLES array, on vertices of cube to avoid plotting fisheyed effect
vert_box=np.array([[box,0,0,0,0,0],[0,box,0,0,0,0],[0,0,box,0,0,0],[box,box,0,0,0,0],[box,box,box,0,0,0],[box,0,box,0,0,0],[0,box,box,0,0,0]])        
partcls=np.row_stack((partcls,vert_box))

#Plotting
fig=plt.figure(figsize=(50,50),dpi=300)
ax=fig.add_subplot(111, projection='3d')
#plt.title('particle Distribution')
scale_mass=partcls[:,4]/sum(partcls[:,4])*100010
vrad=partcls[:,5]**1.1
ax.scatter(partcls[:,0],partcls[:,1],partcls[:,2],c=partcls[:,3]*2,cmap=plt.cm.autumn,s=100)
ax.view_init(elev=0,azim=-90)#upon generating figure, usually have to rotate manually by 90 deg. clockwise 
plt.xlabel('x') 
plt.ylabel('y')
plt.savefig('/import/oth3/ajib0457/bolchoi_z0/investigation/plots/color_slices/bolchoi_halocolor_gd%d_slc%d_thck%sMpc.png' %(grid_den,slc,partcl_thkns))
f.close()
