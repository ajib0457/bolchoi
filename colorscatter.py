import numpy as np
import math as mth
import pandas as pd
#Initial data from density field and classification
grid_nodes=2000
sim_sz=250#Mpc
in_val,fnl_val=-140,140
tot_parts=8     #pertains to eigenvector files, not mass bins
s=3.92
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

for part in range(tot_parts):#here I have to figure out how these have been stored then put them back together, probably just column stack all
#into 1 array
    nrows_in=int(1.*(grid_nodes**3)/tot_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/tot_parts)
    f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/DTFE/files/output_files/eigvecs/fil_recon_vecs_DTFE500_gd%d_smth%sMpc_%d.h5" %(grid_nodes,round(std_dev_phys,3),part), 'r')
    recon_vecs_x[nrows_in:nrows_fn]=f['/group%d/x'%part][:]
    recon_vecs_y[nrows_in:nrows_fn]=f['/group%d/y'%part][:]
    recon_vecs_z[nrows_in:nrows_fn]=f['/group%d/z'%part][:]
    f.close()

mask=(recon_vecs_x ==9)    
recon_vecs_x[mask]=0
mask=(recon_vecs_x >0)
recon_vecs_x[mask]=1
mask=(recon_vecs_x <0)
recon_vecs_x[mask]=1
recon_vecs_x=np.reshape(recon_vecs_x,(grid_nodes,grid_nodes,grid_nodes))

#HALOS ------------
data = pd.read_csv('/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_halos_z0_xyz_m_j',sep=r"\s+",lineterminator='\n', header = None)

data=data.as_matrix()
partcl_500=np.where((data[:,3]/(1.35*10**8))>=500)#filter out halos with <500 particles
data=data[partcl_500]

#Positions
Xc=data[:,0]
Yc=data[:,1]
Zc=data[:,2]
color_cd=np.zeros(len(Xc))
del data

halos=np.column_stack((Xc,Yc,Zc,color_cd))

# -----------------
#pre-binning for Halos ----------
#EXTREMES ALL PARTICLES
#Xc_min=1.0000000000000001e-05
#Xc_max=249.99997999999999
#Yc_min=2.0000000000000002e-05
#Yc_max=249.99996999999999
#Zc_min=1.0000000000000001e-05
#Zc_max=249.99997999999999
#EXTEMES >=500 PARTICLES
Xc_min=0.00134
Xc_max=249.99911adsfa
Yc_min=0.00080999999999999985
Yc_max=249.99973
Zc_min=0.00027
Zc_max=249.99954

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001
#--------------------------------

for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus) 
  
    halos[i,3]=recon_vecs_x[grid_index_x,grid_index_y,grid_index_z]
    
f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/halo_color_500part.h5", 'w')
f.create_dataset('/group/x',data=halos)
f.close()
        
    

    
