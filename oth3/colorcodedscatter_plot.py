import numpy as np
from matplotlib import pyplot as plt
import h5py
import pandas as pd

#Initial data from density field and classification
grid_den=850
sim_sz=250#Mpc
in_val,fnl_val=-140,140
tot_parts=8     #pertains to eigenvector files, not mass bins
s=3.92
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_den#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_den#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

f=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/bolchoi_halo_colordata_grid%s_smth%s_500part_xyzm_id.h5"%(grid_den,round(std_dev_phys,2)), 'r')
data_in=f['/group/x'][:]#already filtered for >=500

rad = pd.read_csv('/import/oth3/ajib0457/bolchoi_z0/catalogs/rockstar/bolchoi_DTFE_rockstar_halos500_m_r',sep=r"\s+",lineterminator='\n', header = None)
rad=rad.as_matrix()#already filtered for >=500

#scale mass and radii of halos
data_in[:,3]=(data_in[:,3]-np.min(data_in[:,3]))/(np.max(data_in[:,3])-np.min(data_in[:,3]))
#data_in[:,3]=data_in[:,3]/(np.sum(data_in[:,3]))
rad[:,1]=(rad[:,1]-np.min(rad[:,1]))/(np.max(rad[:,1])-np.min(rad[:,1]))
#rad[:,1]=rad[:,1]/(np.sum(rad[:,1]))
data=np.column_stack((data_in,rad[:,1]))

#slices options
slc=425#30 slices per 0.1 length
x,y,z=0,1,2#slice through which axis
box=np.max(data[:,0])#subset box length
partcl_thkns=5#Thickness of the particle slice, Mpc
lo_lim_partcl=1.*slc/(grid_den)*box-1.*partcl_thkns/2 #For particle distribution
hi_lim_partcl=lo_lim_partcl+partcl_thkns #For particle distributionn

#Filter particles and halos
partcls=np.array([0,0,0,0,0,0])

for i in range(len(data)):
   
    if (lo_lim_partcl<data[i,y]<hi_lim_partcl):#incremenets are in 0.00333 #wherver slc is, [x,y,z] make sure corresponds with if statement
        #density field
        result_hals=data[i,:]
        partcls=np.row_stack((partcls,result_hals))
partcls = np.delete(partcls, (0), axis=0)

#Plotting
fig, ax = plt.subplots(figsize=(15,15),dpi=400)

i=0#initiate mask for plot loop
for color in ['red', 'green', 'blue','yellow']:
    lss_plt_filt=np.where(partcls[:,4]==i)
    lss=['voids','sheets','filaments','clusters']  
    scale_factor=1000
    ax.scatter(partcls[lss_plt_filt,0],partcls[lss_plt_filt,2],s=partcls[lss_plt_filt,5]*scale_factor,c=color,label=lss[i],alpha=0.9, edgecolors='none')
    i+=1

#ax.view_init(elev=0,azim=-90)#upon generating figure, usually have to rotate manually by 90 deg. clockwise 
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')
ax.legend()
ax.grid(True)
ax.set_xlim([0,250])
ax.set_ylim([0,250])
plt.savefig('/import/oth3/ajib0457/bolchoi_z0/investigation/plots/color_slices/bolchoi_halocolor_gd%d_slc%d_thck%sMpc_scl_vradius.png' %(grid_den,slc,partcl_thkns))
f.close()
