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
slc_min,slc_max=0,170#The length of filament, in terms of slices of sim
trans_min,trans_max=5,75#the translation of centre circle along 1 dimension
s,r=20,20   
a_x_new,a_y_new,a_z_new=fil_isolater(canvasx,canvasy,canvasz,slc_min,slc_max,trans_min,trans_max,s,r,x,y,z) 

vec_mask=np.where(a_x_new!=0)
vecs=np.column_stack((a_x_new[vec_mask].flatten(),a_y_new[vec_mask].flatten(),a_z_new[vec_mask].flatten()))

#Choose random vectors to be dp'd
pluck=np.random.randint(0,len(vecs),size=8000)
b=len(vecs)
pluck=np.unique(pluck)
a=len(pluck)
vecs=vecs[pluck]
#store_spin=np.zeros(int(1.*len(vecs)*(len(vecs)+1)/2))
store_spin=[]
for i in range(len(vecs)):
    for j in range(i+1,len(vecs)):
    
#    spin_dot=np.inner(vecs[i,:],vecs[i+1:len(vecs),:]) #why do i need loop for this
#    store_spin[(len(store_spin)-1)*i:len(vecs)*i+len(vecs)-]=store_spin
        spin_dot=np.inner(vecs[i,:],vecs[j,:]) #why do i need loop for this
        store_spin.append(spin_dot)

store_spin=np.asarray(store_spin) #run when compelte

store_spin=abs(store_spin)
plt.hist(store_spin,bins=700)
plt.xlabel('dot product')
plt.ylabel('qty')

plt.savefig('/import/oth3/ajib0457/bolchoi_z0/investigation/plots/filament_dp/eigvecs_dp_no_of_vecs_sampled%s_tot%s.png' %(a,b))

