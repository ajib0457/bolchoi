import numpy as np
import math as mth
import pandas as pd
import sklearn.preprocessing as skl
import h5py
import pickle
#Initial data from density field and classification
grid_nodes=850
slc=10
slc_thickness=50
box_sz=850
tot_mass_bins=6
mass_bin=5     # 0 to 5
sim_sz=250#Mpc
in_val,fnl_val=-140,140
tot_parts=8     #pertains to eigenvector files, not mass bins
s=3.92
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

recon_vecs_x=np.zeros((grid_nodes**3))
recon_vecs_y=np.zeros((grid_nodes**3))
recon_vecs_z=np.zeros((grid_nodes**3))

for part in range(tot_parts):#here I have to figure out how these have been stored then put them back together, probably just column stack all
#into 1 array
    nrows_in=int(1.*(grid_nodes**3)/tot_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/tot_parts)
    f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d.h5" %(grid_nodes,round(std_dev_phys,3),part), 'r')
    recon_vecs_x[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()

recon_vecs_flt_unnorm=np.column_stack((recon_vecs_x,recon_vecs_y,recon_vecs_z))
del recon_vecs_x
del recon_vecs_y
del recon_vecs_z

recon_vecs_flt_norm=skl.normalize(recon_vecs_flt_unnorm)#I should not normalize becauase they are already normalized and also the classifier (9) mask will be ruined
recon_vecs=np.reshape(recon_vecs_flt_norm,(grid_nodes,grid_nodes,grid_nodes,3))#Three for the 3 vector components
del recon_vecs_flt_norm
recon_vecs_unnorm=np.reshape(recon_vecs_flt_unnorm,(grid_nodes,grid_nodes,grid_nodes,3))#raw eigenvectors along with (9)-filled rows which represent blank vectors
del recon_vecs_flt_unnorm

#HALOS ------------
data = pd.read_csv('/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_halos_z0_xyz_m_j',sep=r"\s+",lineterminator='\n', header = None)

data=data.as_matrix()
#partcl_500=np.where((data[:,3]/(1.35*10**8))>=500)#filter out halos with <500 particles
#data=data[partcl_500]

#Angular momentum
Lx=data[:,4]
Ly=data[:,5]
Lz=data[:,6]
#Positions
Xc=data[:,0]
Yc=data[:,1]
Zc=data[:,2]

del data
#normalized angular momentum vectors v1
halos_mom=np.column_stack((Lx,Ly,Lz))
norm_halos_mom=skl.normalize(halos_mom)
halos=np.column_stack((Xc,Yc,Zc,norm_halos_mom))
# -----------------

#pre-binning for Halos ----------
Xc_min=1.0000000000000001e-05
Xc_max=249.99997999999999
Yc_min=2.0000000000000002e-05
Yc_max=249.99996999999999
Zc_min=1.0000000000000001e-05
Zc_max=249.99997999999999

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001
#--------------------------------

#grid=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_AM_x=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_AM_y=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_AM_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
resid=[]
for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus) 
    if (store_AM_x[grid_index_x,grid_index_y,grid_index_z]==0):#'recon_vecs_unnorm[grid_index_x,grid_index_y,grid_index_z,0]!=9 and' place this within parentheses when filtering out not situated within filament
        store_AM_x[grid_index_x,grid_index_y,grid_index_z]+=halos[i,3]
        store_AM_y[grid_index_x,grid_index_y,grid_index_z]+=halos[i,4]
        store_AM_z[grid_index_x,grid_index_y,grid_index_z]+=halos[i,5]
    else:#'elif (recon_vecs_unnorm[grid_index_x,grid_index_y,grid_index_z,0]!=9):' place this within parentheses when filtering out not situated within filament
        resid.append(np.array([grid_index_x,grid_index_y,grid_index_z,halos[i,3],halos[i,4],halos[i,5]]))
del halos
del recon_vecs_unnorm
del recon_vecs
#store a slice
#store_AM_x=store_AM_x[:,slc-slc_thickness:slc+slc_thickness,:].flatten()
#store_AM_y=store_AM_y[:,slc-slc_thickness:slc+slc_thickness,:].flatten()
#store_AM_z=store_AM_z[:,slc-slc_thickness:slc+slc_thickness,:].flatten()

box_sz_x_min=50
box_sz_x_max=200
box_sz_y_min=200
box_sz_y_max=250
box_sz_z_min=50
box_sz_z_max=100

#store a box
store_AM_x=store_AM_x[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max].flatten()
store_AM_y=store_AM_y[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max].flatten()
store_AM_z=store_AM_z[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max].flatten()

#store resid halos which are overlapping 
resid=np.asarray(resid)
Xc=resid[:,0]
Yc=resid[:,1]
Zc=resid[:,2]
halos=np.column_stack((Xc,Yc,Zc,resid[:,3],resid[:,4],resid[:,5]))
mask=np.zeros(len(resid))
filt_x_min=np.where(Xc>=1.*sim_sz/grid_nodes*box_sz_x_min)
filt_y_min=np.where(Yc>=1.*sim_sz/grid_nodes*box_sz_y_min)
filt_z_min=np.where(Zc>=1.*sim_sz/grid_nodes*box_sz_z_min)
filt_x_max=np.where(Xc<=1.*sim_sz/grid_nodes*box_sz_x_max)
filt_y_max=np.where(Yc<=1.*sim_sz/grid_nodes*box_sz_y_max)
filt_z_max=np.where(Zc<=1.*sim_sz/grid_nodes*box_sz_z_max)
mask[filt_x_min]=1
mask[filt_y_min]+=1
mask[filt_z_min]+=1
mask[filt_x_max]+=1
mask[filt_y_max]+=1
mask[filt_z_max]+=1

halo_filt=(mask==6)
halos=halos[halo_filt]


#f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/AM_situatedingrid%s_slc%s_thcness%s.h5" %(grid_nodes,slc,slc_thickness), 'w')

f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/box_cutout_sz%s_%s_%s_%s_%s_%s_%s_AM_situatedingrid.h5"%(box_sz,box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max), 'w')
f.create_dataset('/AM/x',data=store_AM_x)
f.create_dataset('/AM/y',data=store_AM_y)
f.create_dataset('/AM/z',data=store_AM_z)
f.close()

spin_res=open('/scratch/GAMNSCM2/bolchoi_z0/investigation/resid_grid%s_%s_%s_%s_%s_%s_%s.pkl'%(grid_nodes,box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max),'wb')
#spin_array=open('/project/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/dotproduct/spin_store_%dbins_fnl.pkl'%n_dot_prod,'wb')
pickle.dump(halos,spin_res)
spin_res.close()
