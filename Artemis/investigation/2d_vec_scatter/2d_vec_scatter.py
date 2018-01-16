import numpy as np
from matplotlib import pyplot as plt
import h5py
import sklearn.preprocessing as skl
#Initial data from density field and classification
grid_den=850
sim_sz=250#Mpc
in_val,fnl_val=-140,140
s=3.92
halo_mass_filt=300 
x_cutout=100
z_cutout=100
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_den#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_den#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

f=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/bolchoi_DTFE_rockstar_allhalos_xyz_vxyz_jxyz_m_r.h5", 'r')
data_in=f['/halo'][:]
f.close()
#slices options
slc=4#30 slices per 0.1 length
x,y,z=0,1,2#slice through which axis
box=np.max(data_in[:,0])#subset box length
partcl_thkns=5#Thickness of the particle slice, Mpc
lo_lim_partcl=1.*slc/(grid_den)*box-1.*partcl_thkns/2 #For particle distribution
hi_lim_partcl=lo_lim_partcl+partcl_thkns #For particle distributionn
#Filter halos within slc
mask=np.zeros(len(data_in))
lo_lim_mask=np.where(data_in[:,y]>lo_lim_partcl)
hi_lim_mask=np.where(data_in[:,y]<hi_lim_partcl)
partcl_500=np.where((data_in[:,9]/(1.35*10**8))>halo_mass_filt)#filter out halos with <500 particles
x_mask=np.where(data_in[:,x]<x_cutout)
z_mask=np.where(data_in[:,z]<z_cutout)
mask[lo_lim_mask]=1
mask[hi_lim_mask]+=1
mask[partcl_500]+=1
mask[x_mask]+=1
mask[z_mask]+=1
mask_indx=np.where(mask==5)
catalog_slc=data_in[mask_indx]

#re-normalize projected vectors
catalog_vec=np.column_stack((catalog_slc[:,3],catalog_slc[:,5]))
catalog_vec_norm=skl.normalize(catalog_vec)
#Plot vector field 2d
#for 100x100 slc ,headwidth=10,minshaft=4,linewidth=0.1
fig, ax = plt.subplots(figsize=(10,10),dpi=400)
plt.quiver(catalog_slc[:,0],catalog_slc[:,2],catalog_vec_norm[:,0],catalog_vec_norm[:,1],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
ax.set_xlim([0,x_cutout])
ax.set_ylim([0,z_cutout])
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')

plt.savefig('/import/oth3/ajib0457/bolchoi_z0/investigation/plots/vec_slices/bolchoi_halovec_partcls%s_gd%d_slc%d_thck%sMpc_velocity_yplane_%s_%s.png' %(halo_mass_filt,grid_den,slc,partcl_thkns,x_cutout,z_cutout))
