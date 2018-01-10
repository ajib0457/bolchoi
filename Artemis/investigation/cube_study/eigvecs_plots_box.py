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
box_sz=850
grid_nodes=850
box_sz_x_min=30
box_sz_x_max=220
box_sz_y_min=180
box_sz_y_max=270
box_sz_z_min=30
box_sz_z_max=120

#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys
#These eigenvectors include those of filaments and sheets. Voids and clusters have 0 in them
f=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/eigvecs/my_den_boxcutout_%s_%s_%s_%s_%s_%s_grid%d_smth%sMpc_trial.h5" %(box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max,grid_nodes,round(std_dev_phys,2)), 'r')
recon_vecs_x=f['/slc/x'][:]
recon_vecs_y=f['/slc/y'][:]
recon_vecs_z=f['/slc/z'][:]
f.close()
#This is the mask which contains 0-void, 1-sheet,2-filament and 3-cluster
f1=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/eigvecs/my_den_boxcutout_%s_%s_%s_%s_%s_%s_grid%d_smth%sMpc_trialmask.h5" %(box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max,grid_nodes,round(std_dev_phys,2)), 'r')
mask=f1['/slc/mask'][:]
f1.close()

#these are the binned AM vectors and below are the residual unbinned which were destined to overlap these non-zero pixels
f2=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/box_cutout_sz%s_%s_%s_%s_%s_%s_%s_AM_situatedingrid.h5"%(box_sz,box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max), 'r')
halo_vecs_x=f2['/AM/x'][:]
halo_vecs_y=f2['/AM/y'][:]
halo_vecs_z=f2['/AM/z'][:]
f2.close()
#these are the vectors which are unbinned because I did not want to overplap the ones which are binned (above)
spin_res=open('/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/resid_grid%s_%s_%s_%s_%s_%s_%s.pkl'%(grid_nodes,box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max),'rb')
res=pickle.load(spin_res)

recon_vecs_x=np.reshape(recon_vecs_x,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
recon_vecs_y=np.reshape(recon_vecs_y,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
recon_vecs_z=np.reshape(recon_vecs_z,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))

mask_fil=np.zeros(len(mask))
mask_fil_indx=np.where(mask==2)
mask_fil[mask_fil_indx]=1
mask_fil=np.reshape(mask_fil,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))

mask_sht=np.zeros(len(mask))
mask_sht_indx=np.where(mask==1)
mask_sht[mask_sht_indx]=1
mask_sht=np.reshape(mask_sht,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))

mask=np.reshape(mask,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))

halo_vecs_x=np.reshape(halo_vecs_x,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
halo_vecs_y=np.reshape(halo_vecs_y,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
halo_vecs_z=np.reshape(halo_vecs_z,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))

#3d eigenvector plot using mayavi
mlab.close(all=True)
pltt=mlab.figure(size=(2000,2000))

pltt=mlab.pipeline.vector_field(recon_vecs_x,recon_vecs_y,recon_vecs_z)#optional: vmin=0.6 and vmax=0.4
pltt_halos=mlab.pipeline.vector_field(halo_vecs_x,halo_vecs_y,halo_vecs_z)#optional: vmin=0.6 and vmax=0.4
mlab.pipeline.vectors(pltt,scale_factor=3,mask_points=50,line_width=0.6)#mask_points=2 This is for the eigenvectors lSS
mlab.pipeline.vectors(pltt_halos,scale_factor=1,mask_points=3,line_width=0.6,color=(1,1,1))#This will be for the halo vectors.
#pltt=mlab.pipeline.volume(mlab.pipeline.scalar_field(recon_img),vmin=0.1,vmax=0.9)#optional: vmin=0.6 and vmax=0.4
#mlab.pipeline.vector_cut_plane(pltt,mask_points=20,scale_factor=3.)#sliding slicer
del recon_vecs_x
del recon_vecs_y
del recon_vecs_z
#del halo_vecs_x
#del halo_vecs_y
#del halo_vecs_z
magnitude = mlab.pipeline.extract_vector_norm(pltt) #this line aids in the following line to create isosurfaces.
mlab.pipeline.iso_surface(magnitude, contours=[1.9, 0.5])#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field
#mlab.contour3d(mask_fil)
#mlab.contour3d(mask_sht)
#magnitude_h = mlab.pipeline.extract_vector_norm(pltt_halos) #this line aids in the following line to create isosurfaces.
#mlab.pipeline.iso_surface(magnitude_h, contours=[1.9, 0.5])#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field

#mlab.colorbar(nb_colors=4,nb_labels=0,orientation='vertical')#nb_labels=2,nb_colors=2 to have discrete colors
#mlab.colorbar(orientation='vertical')
mlab.axes()
v=mlab.view(azimuth=-90,elevation=0)#this creates a variable, v, to which mayavi saves the azimuth & elevation angle to

#mlab.savefig('DTFE_gd.png',size=(4000,4000))
