import numpy as np
from mayavi import mlab
import h5py
import sklearn.preprocessing as skl
import matplotlib.pyplot as plt

def fil_isolater(canvasx,canvasy,canvasz,slc_min,slc_max,trans_min,trans_max,s,r,x,y,z):
    '''
    slc_min & slc_max: begin-end of cutout | trans_min,trans_max: translation of circle cnt. | 
    line_l,s: Initial Circle centre | r: radius | m: Path gradient | b: y-intercept of line | fil_rng: ind. variable
    '''
    
    m=-1.*(trans_max-trans_min)/(slc_max-slc_min)#Make sure the gradient is - or +
    b=trans_max-m*slc_min#y intercept of line
    fil_rng=np.arange(slc_min,slc_max,1)#independant variable
    line_l=m*fil_rng+b#y=mx+b
    
    a_x_new=np.zeros((x,y,z))
    a_y_new=np.zeros((x,y,z))
    a_z_new=np.zeros((x,y,z))
    for i in range(slc_max-slc_min):
        for j in range(y):
            for k in range(z):  
                if ((j-line_l[i])**2+(k-s)**2<=r**2):
                    a_x_new[i+slc_min,j,k]=canvasx[i+slc_min,j,k]
                    a_y_new[i+slc_min,j,k]=canvasy[i+slc_min,j,k]
                    a_z_new[i+slc_min,j,k]=canvasz[i+slc_min,j,k]
                    
    return a_x_new,a_y_new,a_z_new
    
#Initial data from density field and classification
sim_sz=250#Mpc
in_val,fnl_val=-140,140
tot_parts=8
s=3.92
box_sz=850
grid_nodes=850
#Sample size STORE THESE BOUNDARIES FOR EVERY NEW SAMPLE YOU ASSESS
box_sz_x_min=0
box_sz_x_max=240
box_sz_y_min=140
box_sz_y_max=290
box_sz_z_min=0
box_sz_z_max=240

#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys
#load all arrays into diction
diction={}
arrys=['recon_vecs_x','recon_vecs_y','recon_vecs_z','mask','Vx','Vy','Vz','Lx','Ly','Lz','Vmass','Vradius','resid']
f=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/my_den_cutout_%s_%s_%s_%s_%s_%s_grid%d_smth%sMpc_eig_xyz_mask_Vxyz_Lxyz_Vm_Vr_resid.h5" %(box_sz_x_min,box_sz_x_max,box_sz_y_min,box_sz_y_max,box_sz_z_min,box_sz_z_max,grid_nodes,round(std_dev_phys,2)), 'r')
for i in arrys:
    diction[i]=f['/%s'%i][:]
f.close()
#Reshape all relevent arrays
arrys=['recon_vecs_x','recon_vecs_y','recon_vecs_z','mask','Vx','Vy','Vz','Lx','Ly','Lz','Vmass','Vradius'] 
for i in arrys:   
    diction[i]=np.reshape(diction[i],(box_sz_x_max-box_sz_x_min,box_sz_y_max-box_sz_y_min,box_sz_z_max-box_sz_z_min))#Reshape all arrays

#Create fake vectors for 'clusters' & 'voids'  
x,y,z=np.shape(diction['mask'])
fakevecs=skl.normalize(np.random.rand(x*y*z,3))
fakevecs_x=np.reshape(fakevecs[:,0],(x,y,z))
fakevecs_y=np.reshape(fakevecs[:,1],(x,y,z))
fakevecs_z=np.reshape(fakevecs[:,2],(x,y,z))
#3d eigenvector plot using mayavi

mlab.close(all=True)
pltt=mlab.figure(size=(2000,2000))

manifest=['filaments','clusters','sheets','voids']#make sure to put 'cluster' before 'filament' in this list otherwise 'filament' will supersede 'cluster' iso
halo_mass_filt=50  #Particle cut-off threshold
for i in manifest:
    #sub-sample. STORE THESE BOUNDARIES FOR EVERY NEW SAMPLE YOU ASSESS
    box_sz_x_min=0
    box_sz_x_max=240
    box_sz_y_min=0
    box_sz_y_max=100
    box_sz_z_min=0
    box_sz_z_max=60
    #Create canvas
    x,y,z=np.shape(diction['mask'])
    canvasx=np.zeros((x,y,z))
    canvasy=np.zeros((x,y,z))
    canvasz=np.zeros((x,y,z))
    if i=='clusters':
        #mask out all but sheet eigenvectors
        mask_indx=np.where(diction['mask']==3)
        canvasx[mask_indx]=fakevecs_x[mask_indx]
        canvasy[mask_indx]=fakevecs_y[mask_indx]
        canvasz[mask_indx]=fakevecs_z[mask_indx]
        
        #Function fil_isolater() inputs
        canvasx=canvasx[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        canvasy=canvasy[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        canvasz=canvasz[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        x,y,z=np.shape(canvasx)#new dimensions of box
        slc_min,slc_max=30,120#The length of filament, in terms of slices of sim
        trans_min,trans_max=40,60#the translation of centre circle along 1 dimension
        s,r=20,15   
        a_x_new,a_y_new,a_z_new=fil_isolater(canvasx,canvasy,canvasz,slc_min,slc_max,trans_min,trans_max,s,r,x,y,z) 
        
        pltt_clus=mlab.pipeline.vector_field(a_x_new,a_y_new,a_z_new)#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_clus,scale_factor=1,mask_points=5000,line_width=0.6,color=(1,0,0))#mask_points=2 This is for the eigenvectors lSS
        magnitude_clus = mlab.pipeline.extract_vector_norm(pltt_clus) #this line aids in the following line to create isosurfaces.
        mlab.pipeline.iso_surface(magnitude_clus, contours=[1.9, 0.5],color=(1,0,0))#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field
#        mlab.axes(ylabel='y[Mpc/h]',xlabel='x[Mpc/h]',zlabel='z[Mpc/h]')
        
    if i=='filaments':
        #mask out all but filament eigenvectors
        mask_indx=np.where(diction['mask']==2)
        canvasx[mask_indx]=diction['recon_vecs_x'][mask_indx]
        canvasy[mask_indx]=diction['recon_vecs_y'][mask_indx]
        canvasz[mask_indx]=diction['recon_vecs_z'][mask_indx]
        
        #Function fil_isolater() inputs
        canvasx=canvasx[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        canvasy=canvasy[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        canvasz=canvasz[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        x,y,z=np.shape(canvasx)#new dimensions of box
        slc_min,slc_max=30,120#The length of filament, in terms of slices of sim
        trans_min,trans_max=40,60#the translation of centre circle along 1 dimension
        s,r=20,15   
        a_x_new,a_y_new,a_z_new=fil_isolater(canvasx,canvasy,canvasz,slc_min,slc_max,trans_min,trans_max,s,r,x,y,z)        
        
        pltt_fil=mlab.pipeline.vector_field(a_x_new,a_y_new,a_z_new)#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_fil,scale_factor=1,mask_points=50,line_width=0.6,color=(1,1,1))#mask_points=2 This is for the eigenvectors lSS
        magnitude_fil = mlab.pipeline.extract_vector_norm(pltt_fil) #this line aids in the following line to create isosurfaces.
        mlab.pipeline.iso_surface(magnitude_fil, contours=[1.9, 0.5],color=(1,1,0),opacity=0.4)#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field
       
    if i=='sheets':
        #mask out all but sheet eigenvectors
        mask_indx=np.where(diction['mask']==1)
        canvasx[mask_indx]=diction['recon_vecs_x'][mask_indx]
        canvasy[mask_indx]=diction['recon_vecs_y'][mask_indx]
        canvasz[mask_indx]=diction['recon_vecs_z'][mask_indx]
        
        #Function fil_isolater() inputs
        canvasx=canvasx[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        canvasy=canvasy[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        canvasz=canvasz[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        x,y,z=np.shape(canvasx)#new dimensions of box
        slc_min,slc_max=0,240#The length of filament, in terms of slices of sim
        trans_min,trans_max=40,60#the translation of centre circle along 1 dimension
        s,r=20,1000000   
        a_x_new,a_y_new,a_z_new=fil_isolater(canvasx,canvasy,canvasz,slc_min,slc_max,trans_min,trans_max,s,r,x,y,z)
        
        pltt_sheet=mlab.pipeline.vector_field(a_x_new,a_y_new,a_z_new)#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_sheet,scale_factor=1,mask_points=200000,line_width=0.6,color=(0.2,0.2,0.9))#mask_points=2 This is for the eigenvectors lSS
        magnitude_sht = mlab.pipeline.extract_vector_norm(pltt_sheet) #this line aids in the following line to create isosurfaces.
        mlab.pipeline.iso_surface(magnitude_sht, contours=[1.9, 0.5],color=(0.2,0.2,0.9),opacity=0.2)#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field
        mlab.axes(ylabel='y[Mpc/h]',xlabel='x[Mpc/h]',zlabel='z[Mpc/h]')

    if i=='voids':
        #mask out all but sheet eigenvectors
        mask_indx=np.where(diction['mask']==0)
        canvasx[mask_indx]=fakevecs_x[mask_indx]
        canvasy[mask_indx]=fakevecs_y[mask_indx]
        canvasz[mask_indx]=fakevecs_z[mask_indx]
        
        #Function fil_isolater() inputs
        canvasx=canvasx[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        canvasy=canvasy[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        canvasz=canvasz[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max]
        x,y,z=np.shape(canvasx)#new dimensions of box
        slc_min,slc_max=30,120#The length of filament, in terms of slices of sim
        trans_min,trans_max=40,60#the translation of centre circle along 1 dimension
        s,r=20,15   
        a_x_new,a_y_new,a_z_new=fil_isolater(canvasx,canvasy,canvasz,slc_min,slc_max,trans_min,trans_max,s,r,x,y,z)
        
        pltt_vod=mlab.pipeline.vector_field(a_x_new,a_y_new,a_z_new)#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_vod,scale_factor=1,mask_points=5000,line_width=0.6,color=(0.06666, 0.06666, 0.1804))#mask_points=2 This is for the eigenvectors lSS
        magnitude_vod = mlab.pipeline.extract_vector_norm(pltt_vod) #this line aids in the following line to create isosurfaces.
        mlab.pipeline.iso_surface(magnitude_vod, contours=[1.9, 0.5],color=(0.06666, 0.06666, 0.1804),name='void')#this is the isosurfaces plotter. You can simply delete these two lines to plot a vector field

    if i=='halos_L':
        #Plot halo AM vectors
        partcl_500=np.where((diction['Vmass']/(1.35*10**8))<halo_mass_filt)#filter out halos with <500 particles
        mask_v=(diction['mask']!=2)
        diction['Lx'][mask_v]=0
        diction['Ly'][mask_v]=0
        diction['Lz'][mask_v]=0
        diction['Lx'][partcl_500]=0
        diction['Ly'][partcl_500]=0
        diction['Lz'][partcl_500]=0
        pltt_fil=mlab.pipeline.vector_field(diction['Lx'][box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max],diction['Ly'][box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max],diction['Lz'][box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max])#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_fil,scale_factor=5,line_width=0.6,color=(0.06666, 0.06666, 0.8))#mask_points=2 This is for the eigenvectors lSS               
        #plot residuals which could not be binned due to overlap issues
        #Resid structure: x(0),y(1),z(2) Vx(3),Vy(4),Vz(5) Lx(6),Ly(7),Lz(8) Vmass(9),Vradius(10)
#        partcl_500=np.where((diction['resid'][:,9]/(1.35*10**8))<halo_mass_filt)
#        diction['resid'][partcl_500]=0
#        for j in range(len(diction['resid'])):
#            x=diction['resid'][j,0]
#            y=diction['resid'][j,1]
#            z=diction['resid'][j,2]
#            if (box_sz_x_min<x<=box_sz_x_max and box_sz_y_min<y<=box_sz_y_max and box_sz_z_min<z<=box_sz_z_max and diction['mask'][x,y,z]==2): 
#                pltt_fil=mlab.pipeline.vector_field(diction['resid'][j,0],diction['resid'][j,1],diction['resid'][j,2],diction['resid'][j,6],diction['resid'][j,7],diction['resid'][j,8])
#                mlab.pipeline.vectors(pltt_fil,scale_factor=3,line_width=0.6,color=(0.06666, 0.06666, 0.8))
                
    if i=='halos_V':
        #Plot halo velocity vectors
        partcl_500=np.where((diction['Vmass']/(1.35*10**8))<halo_mass_filt)#filter out halos with <500 particles
#        mask_v=(diction['mask']!=2)
#        diction['Vx'][mask_v]=0
#        diction['Vy'][mask_v]=0
#        diction['Vz'][mask_v]=0
        diction['Vx'][partcl_500]=0
        diction['Vy'][partcl_500]=0
        diction['Vz'][partcl_500]=0
        pltt_fil=mlab.pipeline.vector_field(diction['Vx'][box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max],diction['Vy'][box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max],diction['Vz'][box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max])#optional: vmin=0.6 and vmax=0.4
        mlab.pipeline.vectors(pltt_fil,scale_factor=20,line_width=2,color=(0,0,0))#mask_points=2 This is for the eigenvectors lSS
        #plot residuals which could not be binned due to overlap issues
        #Resid structure: x(0),y(1),z(2) Vx(3),Vy(4),Vz(5) Lx(6),Ly(7),Lz(8) Vmass(9),Vradius(10)
#        partcl_500=np.where((diction['resid'][:,9]/(1.35*10**8))<halo_mass_filt)                   
#        diction['resid'][partcl_500]=0
#        for j in range(len(diction['resid'])):
#            x=diction['resid'][j,0]
#            y=diction['resid'][j,1]
#            z=diction['resid'][j,2]
#            if (box_sz_x_min<x<=box_sz_x_max and box_sz_y_min<y<=box_sz_y_max and box_sz_z_min<z<=box_sz_z_max and diction['mask'][x,y,z]==2):
#                pltt_fil=mlab.pipeline.vector_field(diction['resid'][j,0],diction['resid'][j,1],diction['resid'][j,2],diction['resid'][j,3],diction['resid'][j,4],diction['resid'][j,5])
#                mlab.pipeline.vectors(pltt_fil,scale_factor=3,line_width=0.6,color=(0.06666, 0.06666, 0.8))   
                     
    if i=='halos_Vradius':
        #Scatter halos with size prop. to Vradius
        canvasx=diction['Vradius']
        pltt_fil=mlab.points3d(canvasx[box_sz_x_min:box_sz_x_max,box_sz_y_min:box_sz_y_max,box_sz_z_min:box_sz_z_max],scale_factor=0.5)



#mlab.axes(ylabel='y[Mpc/h]',xlabel='x[Mpc/h]',zlabel='z[Mpc/h]')
#v=mlab.view(azimuth=270.1,elevation=90,distance=2000)#this creates a variable, v, to which mayavi saves the azimuth & elevation angle to
#mlab.savefig('/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/eigvecs/plots_eigvecs/4slc_gd850_gd.png')
