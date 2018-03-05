import numpy as np
import h5py
from scipy import ndimage

#Initial data from density field and classification
grid_nodes=850
sim_sz=250#Mpc
in_val,fnl_val=-425,425
tot_parts=8     #pertains to eigenvector files, not mass bins
s=11.9
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

recon_vecs_x=np.zeros((grid_nodes**3))
recon_vecs_y=np.zeros((grid_nodes**3))
recon_vecs_z=np.zeros((grid_nodes**3))
mask=np.zeros((grid_nodes**3))
for part in range(tot_parts):

    nrows_in=int(1.*(grid_nodes**3)/tot_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/tot_parts)
    f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/DTFE/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d.h5" %(grid_nodes,round(std_dev_phys,3),part), 'r')
    recon_vecs_x[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
    f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/DTFE/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d_mask.h5" %(grid_nodes,round(std_dev_phys,3),part), 'r')
    mask[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()
    
mask_fil=(mask !=2)
recon_vecs_x[mask_fil]=0
recon_vecs_y[mask_fil]=0
recon_vecs_z[mask_fil]=0

recon_vecs_x=np.reshape(recon_vecs_x,(grid_nodes,grid_nodes,grid_nodes))
recon_vecs_y=np.reshape(recon_vecs_y,(grid_nodes,grid_nodes,grid_nodes))
recon_vecs_z=np.reshape(recon_vecs_z,(grid_nodes,grid_nodes,grid_nodes))

s=1.7
recon_vecs_x_smth = ndimage.gaussian_filter(recon_vecs_x,s,order=0,mode='wrap',truncate=50)
recon_vecs_y_smth = ndimage.gaussian_filter(recon_vecs_y,s,order=0,mode='wrap',truncate=50)
recon_vecs_z_smth = ndimage.gaussian_filter(recon_vecs_z,s,order=0,mode='wrap',truncate=50)

#in terms of pixels
pxl_length_gauss=1.*s/(2*fnl_val)*grid_nodes
#in terms of Mpc/h
std_dev_phys_vec=pxl_length_gauss*(1.*sim_sz/grid_nodes)

f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/DTFE/files/output_files/eigvecs_smthd/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d_vecsmthd_%sMpch.h5" %(grid_nodes,round(std_dev_phys,3),part,std_dev_phys_vec), 'w')
f.create_dataset('/group%d/x_smthd'%part,data=recon_vecs_x_smth.flatten())
f.create_dataset('/group%d/y_smthd'%part,data=recon_vecs_y_smth.flatten())
f.create_dataset('/group%d/z_smthd'%part,data=recon_vecs_z_smth.flatten())
f.close()
