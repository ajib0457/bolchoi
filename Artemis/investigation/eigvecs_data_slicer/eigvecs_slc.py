import numpy as np
import h5py
#Initial data from density field and classification
grid_nodes=850
sim_sz=250#Mpc
in_val,fnl_val=-140,140
tot_parts=8
s=3.92
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

#cutout dimensions, in pixels
box_sz_x_min=0
box_sz_x_max=850
box_sz_y_min=2
box_sz_y_max=6
box_sz_z_min=0
box_sz_z_max=850
diction={}#Dictionary which will hold all arrays within script
arrys=['recon_vecs_x','recon_vecs_y','recon_vecs_z','mask']#every changing list to call desired arrays
for i in arrys:
    diction[i]=np.zeros((grid_nodes**3))

for part in range(tot_parts):#here I have to figure out how these have been stored then put them back together, probably just column stack all
#into 1 array
    nrows_in=int(1.*(grid_nodes**3)/tot_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/tot_parts)
    f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d.h5" %(grid_nodes,round(std_dev_phys,3),part), 'r')
    diction[arrys[0]][nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    diction[arrys[1]][nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    diction[arrys[2]][nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()
    f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d_mask.h5" %(grid_nodes,round(std_dev_phys,3),part), 'r')
    diction[arrys[3]][nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()
    
arrys=['Vx','Vy','Vz','Lx','Ly','Lz','Vmass','Vradius','resid']
f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/bolchoi_grid%s_smth%s_VxyzLxyzVmVr_withresid_situatedingrid.h5"%(grid_nodes,round(std_dev_phys,2)), 'r')
for i in arrys:
    diction[i]=f['/%s'%i][:]
f.close()

arrys=['recon_vecs_x','recon_vecs_y','recon_vecs_z','mask','Vx','Vy','Vz','Lx','Ly','Lz','Vmass','Vradius'] 
for i in arrys:   
    diction[i]=np.reshape(diction[i],(grid_nodes,grid_nodes,grid_nodes))#Reshape all arrays
    diction[i]=diction[i][box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max].flatten()#make cutout of all arrays
       
#Filter out residual halos which are not within box boundaries
arrys=['resid'] 
mask=np.zeros(len(diction[arrys[0]]))
filt_x_min=np.where(diction[arrys[0]][:,0]>=box_sz_x_min)
filt_y_min=np.where(diction[arrys[0]][:,1]>=box_sz_y_min)
filt_z_min=np.where(diction[arrys[0]][:,2]>=box_sz_z_min)
filt_x_max=np.where(diction[arrys[0]][:,0]<=box_sz_x_max)
filt_y_max=np.where(diction[arrys[0]][:,1]<=box_sz_y_max)
filt_z_max=np.where(diction[arrys[0]][:,2]<=box_sz_z_max)
mask[filt_x_min]=1
mask[filt_y_min]+=1
mask[filt_z_min]+=1
mask[filt_x_max]+=1
mask[filt_y_max]+=1
mask[filt_z_max]+=1
halo_filt=(mask==6)
diction[arrys[0]]=diction[arrys[0]][halo_filt]

arrys=['recon_vecs_x','recon_vecs_y','recon_vecs_z','mask','Vx','Vy','Vz','Lx','Ly','Lz','Vmass','Vradius','resid']
f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/my_den_cutout_%s_%s_%s_%s_%s_%s_grid%d_smth%sMpc_eig_xyz_mask_Vxyz_Lxyz_Vm_Vr_resid.h5" %(box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max,grid_nodes,round(std_dev_phys,2)), 'w')
for i in arrys:
    f.create_dataset('/%s'%i,data=diction[i])
f.close()
