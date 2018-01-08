import numpy as np
import sklearn.preprocessing as skl
import math as mth

import h5py
import pandas as pd
grid_nodes=2000
#create random unit vectors
norm_halos_mom=skl.normalize(np.random.rand(grid_nodes**3,3))

f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/AM_situatedingrid.h5" , 'r')
halo_vecs_x=f2['/AM/x'][:]
halo_vecs_y=f2['/AM/y'][:]
halo_vecs_z=f2['/AM/z'][:]
f2.close()

recon_vecs_flt_unnorm=np.column_stack((halo_vecs_x,halo_vecs_y,halo_vecs_z))
del halo_vecs_x
del halo_vecs_y
del halo_vecs_z

recon_vecs_flt_norm=skl.normalize(recon_vecs_flt_unnorm)#I should not normalize becauase they are already normalized and also the classifier (9) mask will be ruined
recon_vecs=np.reshape(recon_vecs_flt_norm,(grid_nodes,grid_nodes,grid_nodes,3))#Three for the 3 vector components
del recon_vecs_flt_norm

data = pd.read_csv('/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_halos_z0_xyz_m_j',sep=r"\s+",lineterminator='\n', header = None)

data=data.as_matrix()
partcl_500=np.where((data[:,3]/(1.35*10**8))>=500)#filter out halos with <500 particles
data=data[partcl_500]

#Angular momentum
Lx=data[:,4]
Ly=data[:,5]
Lz=data[:,6]
#Positions
Xc=data[:,0]
Yc=data[:,1]
Zc=data[:,2]


#normalized angular momentum vectors v1
halos_mom=np.column_stack((Lx,Ly,Lz))
norm_halos_mom=skl.normalize(halos_mom)
halos=np.column_stack((Xc,Yc,Zc,norm_halos_mom))
    
#pre-binning for Halos ----------
Xc_min=np.min(data[:,0])
Xc_max=np.max(data[:,0])
Yc_min=np.min(data[:,1])
Yc_max=np.max(data[:,1])
Zc_min=np.min(data[:,2])
Zc_max=np.max(data[:,2])
del data
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
for i in range(len(halos)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus)
    
    #calculate dot product and bin
    if (recon_vecs[grid_index_x,grid_index_y,grid_index_z,0]!=9):#condition includes recon_vecs_unnorm so that I may normalize the vectors which are being processed
#        print(i)        
        spin_dot=np.inner(halos[i,3:6],recon_vecs[grid_index_x,grid_index_y,grid_index_z,:]) 
        store_spin.append(spin_dot)
store_spin=np.asarray(store_spin)
#print(store_spin)

f1=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/dotprodoutput.h5", 'w')
f1.create_dataset('/AM/d',data=store_spin)
f1.close()
