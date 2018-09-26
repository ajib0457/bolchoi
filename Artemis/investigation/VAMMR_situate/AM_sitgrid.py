import numpy as np
import math as mth
import pandas as pd
import sklearn.preprocessing as skl
import h5py
#Initial data from density field and classification
grid_nodes=850
sim_sz=250#Mpc
in_val,fnl_val=-140,140
tot_parts=8     #pertains to eigenvector files, not mass bins
s=3.92
#Calculate the std deviation in physical units
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

#HALOS ------------
data = pd.read_csv('/scratch/GAMNSCM2/bolchoi_z0/cat_reconfig/files/output_files/bolchoi_DTFE_rockstar_allhalos_xyz_vxyz_jxyz_m_r',sep=r"\s+",lineterminator='\n', header = None)
data=data.as_matrix()

#Positions
Xc=data[:,0]
Yc=data[:,1]
Zc=data[:,2]
#Velocity
Vx=data[:,3]
Vy=data[:,4]
Vz=data[:,5]
#Angular momentum
Lx=data[:,6]
Ly=data[:,7]
Lz=data[:,8]
#virial mass & radius
vmass=data[:,9]
vradius=data[:,10]
vradius_norm=(vradius-np.min(vradius))/(np.max(vradius)-np.min(vradius))
del data
#normalize all vectors
halos_Lxyz=np.column_stack((Lx,Ly,Lz))
norm_halos_Lxyz=skl.normalize(halos_Lxyz)

halos_Vxyz=np.column_stack((Vx,Vy,Vz))
norm_halos_Vxyz=skl.normalize(halos_Vxyz)

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
store_V_x=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_V_y=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_V_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_AM_x=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_AM_y=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_AM_z=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_VM=np.zeros((grid_nodes,grid_nodes,grid_nodes))
store_VR=np.zeros((grid_nodes,grid_nodes,grid_nodes))
resid=[]
for i in range(len(Xc)):
   #Create index related to the eigenvector bins
    grid_index_x=mth.trunc(Xc[i]*Xc_mult-Xc_minus)      
    grid_index_y=mth.trunc(Yc[i]*Yc_mult-Yc_minus) 
    grid_index_z=mth.trunc(Zc[i]*Zc_mult-Zc_minus) 
    if (store_AM_x[grid_index_x,grid_index_y,grid_index_z]==0):#'recon_vecs_unnorm[grid_index_x,grid_index_y,grid_index_z,0]!=9 and' place this within parentheses when filtering out not situated within filament
        
        store_V_x[grid_index_x,grid_index_y,grid_index_z]=norm_halos_Vxyz[i,0]
        store_V_y[grid_index_x,grid_index_y,grid_index_z]=norm_halos_Vxyz[i,1]
        store_V_z[grid_index_x,grid_index_y,grid_index_z]=norm_halos_Vxyz[i,2]        
        store_AM_x[grid_index_x,grid_index_y,grid_index_z]=norm_halos_Lxyz[i,0]
        store_AM_y[grid_index_x,grid_index_y,grid_index_z]=norm_halos_Lxyz[i,1]
        store_AM_z[grid_index_x,grid_index_y,grid_index_z]=norm_halos_Lxyz[i,2]
        store_VM[grid_index_x,grid_index_y,grid_index_z]=vmass[i]
        store_VR[grid_index_x,grid_index_y,grid_index_z]=vradius_norm[i]

    else:#'elif (recon_vecs_unnorm[grid_index_x,grid_index_y,grid_index_z,0]!=9):' place this within parentheses when filtering out not situated within filament
        resid.append(np.array([grid_index_x,grid_index_y,grid_index_z,norm_halos_Vxyz[i,0],norm_halos_Vxyz[i,1],norm_halos_Vxyz[i,2],norm_halos_Lxyz[i,0],norm_halos_Lxyz[i,1],norm_halos_Lxyz[i,2],vmass[i],vradius_norm[i]]))

arrys=['Vx','Vy','Vz','Lx','Ly','Lz','Vmass','Vradius','resid']
f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/investigation/bolchoi_grid%s_smth%s_VxyzLxyzVmVr_withresid_situatedingrid.h5"%(grid_nodes,round(std_dev_phys,2)), 'w')
diction={arrys[0]:store_V_x,arrys[1]:store_V_y,arrys[2]:store_V_z,arrys[3]:store_AM_x,arrys[4]:store_AM_y,arrys[5]:store_AM_z,arrys[6]:store_VM,arrys[7]:store_VR,arrys[8]:resid}
for i in arrys:
    f.create_dataset('/%s'%i,data=diction[i])
f.close()

