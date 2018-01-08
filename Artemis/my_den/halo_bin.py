import numpy as np
import matplotlib
import math as mth
import pickle
import sklearn.preprocessing as skl
import h5py
import pandas as pd
#Initial data from density field and classification
grid_nodes=850
#HALOS ------------

data = pd.read_csv('/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_halos_z0',sep=r"\s+",lineterminator='\n', header = None)

data=data.as_matrix()

data=data.astype(float)

Xc=data[:,0]
Yc=data[:,1]
Zc=data[:,2]
h_mass=data[:,3]
del data
 
halos=np.column_stack((Xc,Yc,Zc))

#pre-binning for Halos ----------
Xc_min=np.min(Xc)
Xc_max=np.max(Xc)
Yc_min=np.min(Yc)
Yc_max=np.max(Yc)
Zc_min=np.min(Zc)
Zc_max=np.max(Zc)

Xc_mult=grid_nodes/(Xc_max-Xc_min)
Yc_mult=grid_nodes/(Yc_max-Yc_min)
Zc_mult=grid_nodes/(Zc_max-Zc_min)

Xc_minus=Xc_min*grid_nodes/(Xc_max-Xc_min)+0.0000001
Yc_minus=Yc_min*grid_nodes/(Yc_max-Yc_min)+0.0000001
Zc_minus=Zc_min*grid_nodes/(Zc_max-Zc_min)+0.0000001
#--------------------------------

#grid=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_spin=np.zeros((grid_nodes,grid_nodes,grid_nodes))
for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(halos[i,0]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(halos[i,1]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(halos[i,2]*Zc_mult-Zc_minus)   
    store_spin[grid_index_x,grid_index_y,grid_index_z]+=h_mass[i] 
store_spin=store_spin.flatten()

f=h5py.File('/scratch/GAMNSCM2/bolchoi_z0/my_den/den_grid%s_halo_bin_bolchoi'%grid_nodes, 'w')
f.create_dataset('/halo',data=store_spin)
f.close()
