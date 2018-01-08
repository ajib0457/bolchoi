import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
import pickle
import pandas as pd
import h5py
#Halos
data = pd.read_csv('/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_halos_z0',sep=r"\s+",lineterminator='\n', header = None)

data=data.as_matrix()

data=data.astype(float)
grid_nodes=850
sim_sz=250 #Mpc
box_sz_x_min=50
box_sz_x_max=200
box_sz_y_min=200
box_sz_y_max=250
box_sz_z_min=50
box_sz_z_max=100
Xc=data[:,0]
Yc=data[:,1]
Zc=data[:,2]
halos=np.column_stack((Xc,Yc,Zc))
mask=np.zeros(len(data))
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

f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/halo_box_cutout_sz%s_%s_%s_%s_%s_%s_grid%s.h5" %(box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max,grid_nodes), 'w')
f.create_dataset('/group/x',data=halos)
f.close()
