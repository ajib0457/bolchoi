import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sympy import *
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib import cm
import math as mth
from mayavi import mlab
import pandas as pd
import h5py
#Initial data from density field and classification
sim_sz=250#Mpc
in_val,fnl_val=-140,140
tot_parts=8
s=3.92
grid_nodes=2000
box_sz=100
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

f=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/dtfe_box/box_sz%s_grid%d_smth%sMpc.h5" %(box_sz,grid_nodes,std_dev_phys), 'r')
recon_vecs_x=f['/slc/x'][:]
recon_vecs_y=f['/slc/y'][:]
recon_vecs_z=f['/slc/z'][:]
f.close()
fil_vecs=np.column_stack((recon_vecs_x,recon_vecs_y,recon_vecs_z))
#these AM are filtered for >=500 particles and also with eigvecs,already normalized
f2=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/dtfe_box/box_cutout_sz%s_AM_situatedingrid.h5"%box_sz, 'r')
halo_vecs_x=f2['/AM/x'][:]
halo_vecs_y=f2['/AM/y'][:]
halo_vecs_z=f2['/AM/z'][:]
f2.close()
halo_vecs=np.column_stack((halo_vecs_x,halo_vecs_y,halo_vecs_z))
#fil_vecs=skl.normalize(fil_vecs)
#halo_vecs=skl.normalize(halo_vecs)
mask_x=np.where(halo_vecs_x!=0)[0]
mask_y=np.where(halo_vecs_y!=0)[0]
mask_z=np.where(halo_vecs_z!=0)[0]

dotprod=np.zeros(len(mask_x))

for i in range(len(mask_x)):
    dotprod[i]=np.inner(halo_vecs[mask_x[i]],fil_vecs[mask_x[i]])
dotprod=abs(dotprod)


spin_res=open('/import/oth3/ajib0457/bolchoi_z0/investigation/dtfe_box/resid.pkl','rb')
resid=pickle.load(spin_res)
resid=np.asarray(resid)
mask=np.zeros(len(resid))
a=np.where(resid[:,0]<=100)
b=np.where(resid[:,1]<=100)
c=np.where(resid[:,2]<=100)
mask[a]=1
mask[b]+=1
mask[c]+=1

resid_cube100=resid[np.where(mask==3)]
dotprod_last=np.inner(resid_cube100[:,3:6],fil_vecs[mask_x[52]])
dotprod=np.append(dotprod,dotprod_last)
print(dotprod)
print(len(dotprod))
