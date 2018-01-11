import numpy as np
import pickle
from mayavi import mlab
import h5py
import sklearn.preprocessing as skl

#Initial data from density field and classification
sim_sz=250#Mpc
in_val,fnl_val=-140,140
tot_parts=8
s=3.92
box_sz=850
grid_nodes=850
#cutout size
box_sz_x_min=0
box_sz_x_max=850
box_sz_y_min=2
box_sz_y_max=6
box_sz_z_min=0
box_sz_z_max=850

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
recon_vecs_x=np.reshape(recon_vecs_x,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
recon_vecs_y=np.reshape(recon_vecs_y,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
recon_vecs_z=np.reshape(recon_vecs_z,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))

#This is the mask which contains 0-void, 1-sheet,2-filament and 3-cluster
f1=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/eigvecs/my_den_boxcutout_%s_%s_%s_%s_%s_%s_grid%d_smth%sMpc_trialmask.h5" %(box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max,grid_nodes,round(std_dev_phys,2)), 'r')
mask=f1['/slc/mask'][:]
f1.close()
mask=np.reshape(mask,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
'''
#these are the binned AM vectors and below are the residual unbinned which were destined to overlap these non-zero pixels
f2=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/box_cutout_sz%s_%s_%s_%s_%s_%s_%s_AM_situatedingrid.h5"%(box_sz,box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max), 'r')
halo_vecs_x=f2['/AM/x'][:]
halo_vecs_y=f2['/AM/y'][:]
halo_vecs_z=f2['/AM/z'][:]
f2.close()
halo_vecs_x=np.reshape(halo_vecs_x,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
halo_vecs_y=np.reshape(halo_vecs_y,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
halo_vecs_z=np.reshape(halo_vecs_z,(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))
#these are the vectors which are unbinned because I did not want to overplap the ones which are binned (above)
spin_res=open('/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/resid_grid%s_%s_%s_%s_%s_%s_%s.pkl'%(grid_nodes,box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max),'rb')
res=pickle.load(spin_res)
'''
#new plotting idea
x,y,z=np.shape(mask)
fakevecs=skl.normalize(np.random.rand(x*y*z,3))
#fakevecs=np.reshape(fakevecs,(x,y,z,3))#Fake vectors to create isosurfaces for each LSS
fakevecs_x=np.reshape(fakevecs[:,0],(x,y,z))
fakevecs_y=np.reshape(fakevecs[:,1],(x,y,z))
fakevecs_z=np.reshape(fakevecs[:,2],(x,y,z))
#3d eigenvector plot using mayavi
sub_slc=2
post_slc=3
lim_x_min=0
lim_x_max=850
lim_z_min=0
lim_z_max=850

mlab.close(all=True)
pltt=mlab.figure(size=(2000,2000))

manifest=['clusters','filaments','sheets','voids']
for i in manifest:
    
    if i=='clusters':
        #mask out all but sheet eigenvectors
        recon_vecs_x_clus=np.zeros((x,y,z))
        recon_vecs_y_clus=np.zeros((x,y,z))
        recon_vecs_z_clus=np.zeros((x,y,z))
        mask_indx=np.where(mask==3)
        recon_vecs_x_clus[mask_indx]=fakevecs_x[mask_indx]
        recon_vecs_y_clus[mask_indx]=fakevecs_y[mask_indx]
        recon_vecs_z_clus[mask_indx]=fakevecs_z[mask_indx]        
        pltt_clus=mlab.pipeline.vector_field(recon_vecs_x_clus[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max],recon_vecs_y_clus[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max],recon_vecs_z_clus[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max])#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_clus,scale_factor=3,mask_points=1,line_width=0.6,color=(1,0,0))#mask_points=2 This is for the eigenvectors lSS
        magnitude_clus = mlab.pipeline.extract_vector_norm(pltt_clus) #this line aids in the following line to create isosurfaces.
        mlab.pipeline.iso_surface(magnitude_clus, contours=[1.9, 0.5],color=(1,0,0))#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field
        
    if i=='filaments':
        #mask out all but filament eigenvectors
        recon_vecs_x_fil=np.zeros((x,y,z))
        recon_vecs_y_fil=np.zeros((x,y,z))
        recon_vecs_z_fil=np.zeros((x,y,z))
        mask_indx=np.where(mask==2)
        recon_vecs_x_fil[mask_indx]=recon_vecs_x[mask_indx]
        recon_vecs_y_fil[mask_indx]=recon_vecs_y[mask_indx]
        recon_vecs_z_fil[mask_indx]=recon_vecs_z[mask_indx]
        pltt_fil=mlab.pipeline.vector_field(recon_vecs_x_fil[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max],recon_vecs_y_fil[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max],recon_vecs_z_fil[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max])#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_fil,scale_factor=5,mask_points=20,line_width=0.6,color=(1,1,0))#mask_points=2 This is for the eigenvectors lSS
        magnitude_fil = mlab.pipeline.extract_vector_norm(pltt_fil) #this line aids in the following line to create isosurfaces.
        mlab.pipeline.iso_surface(magnitude_fil, contours=[1.9, 0.5],color=(1,1,0))#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field
       
    if i=='sheets':
        #mask out all but sheet eigenvectors
        recon_vecs_x_sht=np.zeros((x,y,z))
        recon_vecs_y_sht=np.zeros((x,y,z))
        recon_vecs_z_sht=np.zeros((x,y,z))
        mask_indx=np.where(mask==1)
        recon_vecs_x_sht[mask_indx]=recon_vecs_x[mask_indx]
        recon_vecs_y_sht[mask_indx]=recon_vecs_y[mask_indx]
        recon_vecs_z_sht[mask_indx]=recon_vecs_z[mask_indx]
        pltt_sheet=mlab.pipeline.vector_field(recon_vecs_x_sht[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max],recon_vecs_y_sht[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max],recon_vecs_z_sht[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max])#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_sheet,scale_factor=5,mask_points=20,line_width=0.6,color=(0.2,0.2,0.9))#mask_points=2 This is for the eigenvectors lSS
        magnitude_sht = mlab.pipeline.extract_vector_norm(pltt_sheet) #this line aids in the following line to create isosurfaces.
        mlab.pipeline.iso_surface(magnitude_sht, contours=[1.9, 0.5],color=(0.2,0.2,0.9))#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field
        
    if i=='voids':
        #mask out all but sheet eigenvectors
        recon_vecs_x_vd=np.zeros((x,y,z))
        recon_vecs_y_vd=np.zeros((x,y,z))
        recon_vecs_z_vd=np.zeros((x,y,z))
        mask_indx=np.where(mask==0)
        recon_vecs_x_vd[mask_indx]=fakevecs_x[mask_indx]
        recon_vecs_y_vd[mask_indx]=fakevecs_y[mask_indx]
        recon_vecs_z_vd[mask_indx]=fakevecs_z[mask_indx]
        pltt_vod=mlab.pipeline.vector_field(recon_vecs_x_vd[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max],recon_vecs_y_vd[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max],recon_vecs_z_vd[lim_x_min:lim_x_max,sub_slc:post_slc,lim_z_min:lim_z_max])#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_vod,scale_factor=5,mask_points=20,line_width=0.6,color=(0.06666, 0.06666, 0.1804))#mask_points=2 This is for the eigenvectors lSS
        magnitude_vod = mlab.pipeline.extract_vector_norm(pltt_vod) #this line aids in the following line to create isosurfaces.
        mlab.pipeline.iso_surface(magnitude_vod, contours=[1.9, 0.5],color=(0.06666, 0.06666, 0.1804),name='void')#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field

mlab.axes(z_axis_visibility=False,ranges=[0,250,0,250,0,250],xlabel='x[Mpc/h]',zlabel='y[Mpc/h]')
v=mlab.view(azimuth=270.1,elevation=90,distance=2000)#this creates a variable, v, to which mayavi saves the azimuth & elevation angle to
mlab.savefig('/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/eigvecs/plots_eigvecs/4slc_gd850_gd.png')
