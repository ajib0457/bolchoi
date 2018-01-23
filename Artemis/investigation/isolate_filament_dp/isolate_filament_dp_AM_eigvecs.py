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

#Bring in the eigvecs------------------------------------------------------------   
sub_box_sz_x_min=0
sub_box_sz_x_max=240
sub_box_sz_y_min=0
sub_box_sz_y_max=100
sub_box_sz_z_min=0
sub_box_sz_z_max=60
#Create canvas
x,y,z=np.shape(diction['mask'])
canvasx=np.zeros((x,y,z))
canvasy=np.zeros((x,y,z))
canvasz=np.zeros((x,y,z))

#mask out all but filament eigenvectors
mask_indx=np.where(diction['mask']==2)
canvasx[mask_indx]=diction['recon_vecs_x'][mask_indx]
canvasy[mask_indx]=diction['recon_vecs_y'][mask_indx]
canvasz[mask_indx]=diction['recon_vecs_z'][mask_indx]

#Function fil_isolater() inputs
canvasx=canvasx[sub_box_sz_x_min:sub_box_sz_x_max,sub_box_sz_y_min:sub_box_sz_y_max,sub_box_sz_z_min:sub_box_sz_z_max]
canvasy=canvasy[sub_box_sz_x_min:sub_box_sz_x_max,sub_box_sz_y_min:sub_box_sz_y_max,sub_box_sz_z_min:sub_box_sz_z_max]
canvasz=canvasz[sub_box_sz_x_min:sub_box_sz_x_max,sub_box_sz_y_min:sub_box_sz_y_max,sub_box_sz_z_min:sub_box_sz_z_max]
x,y,z=np.shape(canvasx)#new dimensions of box
slc_min,slc_max=0,170#The length of filament, in terms of slices of sim
trans_min,trans_max=5,75#the translation of centre circle along 1 dimension
s,r=20,20   
a_x_new_eigvecs,a_y_new_eigvecs,a_z_new_eigvecs=fil_isolater(canvasx,canvasy,canvasz,slc_min,slc_max,trans_min,trans_max,s,r,x,y,z)

#bring in the AM vectors--------------------------------------------------------
halo_mass_filt=500
partcl_500=np.where((diction['Vmass']/(1.35*10**8))<halo_mass_filt)#filter out halos with <500 particles
mask_v=(diction['mask']!=2)
diction['Lx'][mask_v]=0
diction['Ly'][mask_v]=0
diction['Lz'][mask_v]=0
diction['Lx'][partcl_500]=0
diction['Ly'][partcl_500]=0
diction['Lz'][partcl_500]=0

canvasx= diction['Lx']
canvasy= diction['Ly']
canvasz= diction['Lz']

#Function fil_isolater() inputs
canvasx=canvasx[sub_box_sz_x_min:sub_box_sz_x_max,sub_box_sz_y_min:sub_box_sz_y_max,sub_box_sz_z_min:sub_box_sz_z_max]
canvasy=canvasy[sub_box_sz_x_min:sub_box_sz_x_max,sub_box_sz_y_min:sub_box_sz_y_max,sub_box_sz_z_min:sub_box_sz_z_max]
canvasz=canvasz[sub_box_sz_x_min:sub_box_sz_x_max,sub_box_sz_y_min:sub_box_sz_y_max,sub_box_sz_z_min:sub_box_sz_z_max]
x,y,z=np.shape(canvasx)#new dimensions of box
slc_min,slc_max=0,170#The length of filament, in terms of slices of sim
trans_min,trans_max=5,75#the translation of centre circle along 1 dimension
s,r=20,20   
a_x_new_AM,a_y_new_AM,a_z_new_AM=fil_isolater(canvasx,canvasy,canvasz,slc_min,slc_max,trans_min,trans_max,s,r,x,y,z)

#Now bring in the residuals -------------------------------------------------------
#Resid structure: x(0),y(1),z(2) Vx(3),Vy(4),Vz(5) Lx(6),Ly(7),Lz(8) Vmass(9),Vradius(10)
partcl_500=np.where((diction['resid'][:,9]/(1.35*10**8))>halo_mass_filt)
filt_x_min=np.where(diction['resid'][:,0]>sub_box_sz_x_min)#THESE WILL CHANGE FOR EACH SUBSAMPLE
filt_y_min=np.where(diction['resid'][:,1]>box_sz_y_min)#THESE WILL CHANGE FOR EACH SUBSAMPLE
filt_z_min=np.where(diction['resid'][:,2]>sub_box_sz_z_min)#THESE WILL CHANGE FOR EACH SUBSAMPLE
filt_x_max=np.where(diction['resid'][:,0]<sub_box_sz_x_max)#THESE WILL CHANGE FOR EACH SUBSAMPLE
filt_y_max=np.where(diction['resid'][:,1]<box_sz_y_min+sub_box_sz_y_max)#THESE WILL CHANGE FOR EACH SUBSAMPLE
filt_z_max=np.where(diction['resid'][:,2]<sub_box_sz_z_max)#THESE WILL CHANGE FOR EACH SUBSAMPLE
mask=np.zeros(len(diction['resid']))
mask[filt_x_min]=1
mask[filt_y_min]+=1
mask[filt_z_min]+=1
mask[filt_x_max]+=1
mask[filt_y_max]+=1
mask[filt_z_max]+=1
mask[partcl_500]+=1
sub_smpl=np.asarray(np.where(mask==7)).flatten()
resid_AM=[]
for j in sub_smpl:
    x=diction['resid'][j,0]
    y=diction['resid'][j,1]
    z=diction['resid'][j,2]
    #This if statement used to isolate filament within subsample 
    if (a_x_new_eigvecs[int(x),int(y-box_sz_y_min),int(z)]!=0):#THESE WILL CHANGE FOR EACH SUBSAMPLE
        resid_AM.append(np.array([x,y-box_sz_y_min,z,diction['resid'][j,6],diction['resid'][j,7],diction['resid'][j,8]]))
resid_AM=np.asarray(resid_AM)

#now dp main AM with eigvecs-----------------------------------------------------------        
dp_AMvecs_indx=np.asarray(np.where(a_x_new_AM!=0)).transpose()
dp_vals=[]
for i in dp_AMvecs_indx:
    #pull out AM vecs
    AM_vec=np.column_stack((a_x_new_AM[i[0],i[1],i[2]],a_y_new_AM[i[0],i[1],i[2]],a_z_new_AM[i[0],i[1],i[2]]))
    eigvec_vec=np.column_stack((a_x_new_eigvecs[i[0],i[1],i[2]],a_y_new_eigvecs[i[0],i[1],i[2]],a_z_new_eigvecs[i[0],i[1],i[2]]))    
    dp=np.inner(AM_vec,eigvec_vec)
    dp_vals.append(dp)

#now dp residuals with eigvecs--------------------------------------------------------
for i in range(len(resid_AM)):
    AM_vec=resid_AM[i,3:6]
    eigvec_vec=np.column_stack((a_x_new_eigvecs[int(resid_AM[i,0]),int(resid_AM[i,1]),int(resid_AM[i,2])],a_y_new_eigvecs[int(resid_AM[i,0]),int(resid_AM[i,1]),int(resid_AM[i,2])],a_z_new_eigvecs[int(resid_AM[i,0]),int(resid_AM[i,1]),int(resid_AM[i,2])]))    
    dp=np.inner(AM_vec,eigvec_vec)
    dp_vals.append(dp)
dp_vals=np.asarray(dp_vals)

#Now plot just for scrutiny

store_spin=abs(dp_vals)
bins=15
data=np.histogram(store_spin,bins=bins,density=True)
bin_vals=np.delete(data[1],len(data[1])-1,0)
figure=plt.figure(figsize=(8,5),dpi=50)
plt.plot(bin_vals,data[0],color='r',label='data')
plt.axhline(y=1, xmin=0, xmax=15, color = 'k',linestyle='--')
#plt.hist(store_spin,bins=700)
plt.xlabel('dot product')
plt.ylabel('Q t y')

plt.savefig('/import/oth3/ajib0457/bolchoi_z0/investigation/plots/filament_dp/dp_AM-eigvecs_filament_two_normed.png')
     
