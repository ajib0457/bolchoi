import numpy as np
import h5py
#Initial data from density field and classification
grid_nodes=850
sim_sz=250#Mpc
in_val,fnl_val=-140,140
tot_parts=8
s=3.92
slc=15
slc_thickness=10
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

recon_vecs_x=np.zeros((grid_nodes**3))
recon_vecs_y=np.zeros((grid_nodes**3))
recon_vecs_z=np.zeros((grid_nodes**3))
mask=np.zeros((grid_nodes**3))

for part in range(tot_parts):#here I have to figure out how these have been stored then put them back together, probably just column stack all
#into 1 array
    nrows_in=int(1.*(grid_nodes**3)/tot_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/tot_parts)
    f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d.h5" %(grid_nodes,round(std_dev_phys,3),part), 'r')
    recon_vecs_x[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
    f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d_mask.h5" %(grid_nodes,round(std_dev_phys,3),part), 'r')
    mask[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()

recon_vecs_x=np.reshape(recon_vecs_x,(grid_nodes,grid_nodes,grid_nodes))
recon_vecs_y=np.reshape(recon_vecs_y,(grid_nodes,grid_nodes,grid_nodes))
recon_vecs_z=np.reshape(recon_vecs_z,(grid_nodes,grid_nodes,grid_nodes))
mask=np.reshape(mask,(grid_nodes,grid_nodes,grid_nodes))
box_sz_x_min=50
box_sz_x_max=200
box_sz_y_min=200
box_sz_y_max=250
box_sz_z_min=50
box_sz_z_max=100

#FUll slice
recon_vecs_x=recon_vecs_x[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max].flatten()    
recon_vecs_y=recon_vecs_y[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max].flatten()
recon_vecs_z=recon_vecs_z[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max].flatten()
mask=mask[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max].flatten()
#recon_vecs_x=recon_vecs_x[:,slc-slc_thickness:slc+slc_thickness,:].flatten()
#recon_vecs_y=recon_vecs_y[:,slc-slc_thickness:slc+slc_thickness,:].flatten()
#recon_vecs_z=recon_vecs_z[:,slc-slc_thickness:slc+slc_thickness,:].flatten()

f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/my_den_boxcutout_%s_%s_%s_%s_%s_%s_grid%d_smth%sMpc_trial.h5" %(box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max,grid_nodes,round(std_dev_phys,2)), 'w')
#f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/box_sz%s_grid%d_smth%sMpc.h5" %(box_sz,grid_nodes,round(std_dev_phys,2)), 'w')
f.create_dataset('/slc/x',data=recon_vecs_x)
f.create_dataset('/slc/y',data=recon_vecs_y)
f.create_dataset('/slc/z',data=recon_vecs_z)
f.close()

f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/my_den_boxcutout_%s_%s_%s_%s_%s_%s_grid%d_smth%sMpc_trialmask.h5" %(box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max,grid_nodes,round(std_dev_phys,2)), 'w')
f2.create_dataset('/slc/mask',data=mask)
f2.close()
