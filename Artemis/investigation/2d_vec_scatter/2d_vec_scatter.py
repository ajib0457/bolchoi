import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import sklearn.preprocessing as skl
import h5py
import math as mth
#Initial data from density field and classification
grid_den=850
sim_sz=250#Mpc
in_val,fnl_val=-140,140
s=3.92
halo_mass_filt=500 
x_cutout=100
z_cutout=100
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_den#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_den#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

#import eigenvector fialed and mask
tot_parts=8
mask=np.zeros((grid_den**3))
recon_vecs_x=np.zeros((grid_den**3))
recon_vecs_y=np.zeros((grid_den**3))
recon_vecs_z=np.zeros((grid_den**3))
for part in range(tot_parts):
#into 1 array
    nrows_in=int(1.*(grid_den**3)/tot_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_den**3)/tot_parts)
    f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/DTFE/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d.h5" %(grid_den,round(std_dev_phys,3),part), 'r')
    recon_vecs_x[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
    f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/DTFE/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d_mask.h5" %(grid_den,round(std_dev_phys,3),part), 'r')
    mask[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()
mask=np.reshape(mask,(grid_den,grid_den,grid_den))#density field
recon_vecs_x=np.reshape(recon_vecs_x,(grid_den,grid_den,grid_den))#density field
recon_vecs_y=np.reshape(recon_vecs_y,(grid_den,grid_den,grid_den))#density field
recon_vecs_z=np.reshape(recon_vecs_z,(grid_den,grid_den,grid_den))#density field

f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_allhalos_xyz_vxyz_jxyz_m_r.h5", 'r')
data_in=f['/halo'][:]#x(0),y(1),z(2) Vx(3),Vy(4),Vz(5) Lx(6),Ly(7),Lz(8) Vmass(9),Vradius(10)
f.close()

#slices options
slc=425#30 slices per 0.1 length
x,y,z=0,1,2#slice through which axis
box=np.max(data_in[:,0])#subset box length
partcl_thkns=5#Thickness of the particle slice, Mpc
lo_lim_partcl=1.*slc/(grid_den)*box-1.*partcl_thkns/2 #For particle distribution
hi_lim_partcl=lo_lim_partcl+partcl_thkns #For particle distributionn
#Filter halos within slc
mask_halos=np.zeros(len(data_in))
lo_lim_mask=np.where(data_in[:,y]>lo_lim_partcl)
hi_lim_mask=np.where(data_in[:,y]<hi_lim_partcl)
partcl_500=np.where((data_in[:,9]/(1.35*10**8))>halo_mass_filt)#filter out halos with <500 particles
x_mask=np.where(data_in[:,x]<x_cutout)
z_mask=np.where(data_in[:,z]<z_cutout)
mask_halos[lo_lim_mask]=1
mask_halos[hi_lim_mask]+=1
mask_halos[partcl_500]+=1
mask_halos[x_mask]+=1
mask_halos[z_mask]+=1
mask_indx=np.where(mask_halos==5)
catalog_slc=data_in[mask_indx]

#Now filter out relevent eigvecs
Xc_min=1.0000000000000001e-05
Xc_max=249.99997999999999
Yc_min=2.0000000000000002e-05
Yc_max=249.99996999999999
Zc_min=1.0000000000000001e-05
Zc_max=249.99997999999999

Xc_mult=grid_den/(Xc_max-Xc_min)
Yc_mult=grid_den/(Yc_max-Yc_min)
Zc_mult=grid_den/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_den/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_den/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_den/(Zc_max-Zc_min)+0.0000001
#--------------------------------

fnl_halos_vecs=[]
for i in range(len(catalog_slc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(catalog_slc[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(catalog_slc[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(catalog_slc[i,2]*Zc_mult-Zc_minus) 
    #calculate dot product and bin
    if (mask[grid_index_x,grid_index_y,grid_index_z]==2):#condition includes recon_vecs_unnorm so that I may normalize the vectors which are being processed
        fnl_halos_vecs.append(np.hstack((catalog_slc[i],recon_vecs_x[grid_index_x,grid_index_y,grid_index_z],recon_vecs_y[grid_index_x,grid_index_y,grid_index_z],recon_vecs_z[grid_index_x,grid_index_y,grid_index_z])))
fnl_halos_vecs=np.asarray(fnl_halos_vecs)

#Plot
fig, ax = plt.subplots(figsize=(10,30),dpi=350)

#re-normalize projected vectors
catalog_vec=np.column_stack((fnl_halos_vecs[:,3],fnl_halos_vecs[:,5]))
catalog_vec_norm=skl.normalize(catalog_vec)
#Plot velocity vec field 2d
ax=plt.subplot2grid((3,1), (1,0))
plt.quiver(catalog_slc[:,0],catalog_slc[:,2],catalog_vec_norm[:,0],catalog_vec_norm[:,1],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
ax.set_xlim([0,x_cutout])
ax.set_ylim([0,z_cutout])
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')
plt.title('velocity vectors')
catalog_vec=np.column_stack((fnl_halos_vecs[:,6],fnl_halos_vecs[:,8]))
catalog_vec_norm=skl.normalize(catalog_vec)
#Plot AM field 2d
ax=plt.subplot2grid((3,1), (0,0))
plt.quiver(catalog_slc[:,0],catalog_slc[:,2],catalog_vec_norm[:,0],catalog_vec_norm[:,1],headwidth=15,minshaft=9,linewidth=0.07,scale=40)
ax.set_xlim([0,x_cutout])
ax.set_ylim([0,z_cutout])
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')
plt.title('Angular Momentum vectors')
catalog_vec=np.column_stack((fnl_halos_vecs[:,11],fnl_halos_vecs[:,13]))
catalog_vec_norm=skl.normalize(catalog_vec)
#Plot eigvecs field 2d
ax=plt.subplot2grid((3,1), (2,0))
plt.quiver(catalog_slc[:,0],catalog_slc[:,2],catalog_vec_norm[:,0],catalog_vec_norm[:,1],headwidth=0,minshaft=9,linewidth=0.07,scale=40)
ax.set_xlim([0,x_cutout])
ax.set_ylim([0,z_cutout])
plt.xlabel('x[Mpc/h]') 
plt.ylabel('y[Mpc/h]')
plt.title('filament axis')

plt.savefig('/scratch/GAMNSCM2/bolchoi_z0/investigation/bolchoi_halovec_partcls%s_gd%d_slc%d_thck%sMpc_vel_AM_yplane_%s_%s.png' %(halo_mass_filt,grid_den,slc,partcl_thkns,x_cutout,z_cutout))
