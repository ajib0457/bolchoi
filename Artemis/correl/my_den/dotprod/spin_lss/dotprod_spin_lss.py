import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import math as mth
import pandas as pd
import pickle
import sklearn.preprocessing as skl
import h5py
#Initial data from density field and classification
grid_nodes=850
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

for part in range(tot_parts):
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
# -----------------

#HALOS ------------
data = pd.read_csv('/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_halos_z0_xyz_m_j',sep=r"\s+",lineterminator='\n', header = None)

data=data.as_matrix()
partcl_500=np.where((data[:,3]/(1.35*10**8))>=500)#filter out halos with <500 particles
data=data[partcl_500]

halo_mass=data[:,3]
log_halo_mass=np.log10(halo_mass)#convert into log(M)
mass_intvl=(np.max(log_halo_mass)-np.min(log_halo_mass))/tot_mass_bins
#del halo_mass

low_int_mass=np.min(log_halo_mass)+mass_intvl*mass_bin
hi_int_mass=low_int_mass+mass_intvl
mass_mask=np.zeros(len(log_halo_mass))
loint=np.where(log_halo_mass>=low_int_mass)#Change these two numbers as according to the above intervals
hiint=np.where(log_halo_mass<hi_int_mass)#Change these two numbers as according to the above intervals
mass_mask[loint]=1
mass_mask[hiint]=mass_mask[hiint]+1
mass_indx=np.where(mass_mask==2)

#Angular momentum
Lx=data[:,4]
Ly=data[:,5]
Lz=data[:,6]
#Positions
Xc=data[:,0]
Xc=Xc[mass_indx]
Yc=data[:,1]
Yc=Yc[mass_indx]
Zc=data[:,2]
Zc=Zc[mass_indx]
del data
#normalized angular momentum vectors v1
halos_mom=np.column_stack((Lx,Ly,Lz))

halos_mom=halos_mom[mass_indx]
norm_halos_mom=skl.normalize(halos_mom)
#del halos_mom 
halos=np.column_stack((Xc,Yc,Zc,norm_halos_mom))

#del norm_halos_mom
# -----------------

#pre-binning for Halos ----------
#EXTREMES ALL PARTICLES
Xc_min=1.0000000000000001e-05
Xc_max=249.99997999999999
Yc_min=2.0000000000000002e-05
Yc_max=249.99996999999999
Zc_min=1.0000000000000001e-05
Zc_max=249.99997999999999
#EXTEMES >=500 PARTICLES
#Xc_min=0.00134
#Xc_max=249.99911
#Yc_min=0.00080999999999999985
#Yc_max=249.99973
#Zc_min=0.00027
#Zc_max=249.99954


Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001
#--------------------------------

#dot product details
n_dot_prod=0

#grid=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_spin=[]
for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus) 
    #calculate dot product and bin
    if (recon_vecs_unnorm[grid_index_x,grid_index_y,grid_index_z,0]!=9):#condition includes recon_vecs_unnorm so that I may normalize the vectors which are being processed
        spin_dot=np.inner(halos[i,3:6],recon_vecs[grid_index_x,grid_index_y,grid_index_z,:]) 
        store_spin.append(spin_dot)
del recon_vecs_unnorm
del recon_vecs
del halos
store_spin=np.asarray(store_spin)      
     
spin_array=open('/project/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/dotproduct/spin_lss/DTFE_grid%d_spin_store_%dbins_fil_Log%s-%s_smth%sMpc_%sbins.pkl'%(grid_nodes,n_dot_prod,round(low_int_mass,2),round(hi_int_mass,2),round(std_dev_phys,3),tot_mass_bins),'wb')
#spin_array=open('/project/GAMNSCM2/snapshot_012_LSS_class/correl/DTFE/files/output_files/dotproduct/spin_store_%dbins_fnl.pkl'%n_dot_prod,'wb')
pickle.dump(store_spin,spin_array)
spin_array.close()
