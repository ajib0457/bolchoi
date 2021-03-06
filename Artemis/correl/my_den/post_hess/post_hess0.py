import numpy as np
from sympy import *
import pickle
from matplotlib import cm
import pandas as pd
import h5py
import sklearn.preprocessing as skl
#I should make 5 codes to calculate
#225GB+
part=0#Change for each file. START FROM 0

tot_parts=8
sim_sz=250#Mpc 
in_val,fnl_val=-140,140
s=3.92
grid_nodes=850
nrows_in=int(1.*(grid_nodes**3)/tot_parts*part)
nrows_fn=nrows_in+int(1.*(grid_nodes**3)/tot_parts)
grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
std_dev_phys=1.*s/val_phys*grid_phys

f1=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%sMpc_ifft_dxx.h5" %(grid_nodes,round(std_dev_phys,3)), 'r')
f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%sMpc_ifft_dxy.h5" %(grid_nodes,round(std_dev_phys,3)), 'r')
f3=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%sMpc_ifft_dyy.h5" %(grid_nodes,round(std_dev_phys,3)), 'r')
f4=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%sMpc_ifft_dzz.h5" %(grid_nodes,round(std_dev_phys,3)), 'r')
f5=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%sMpc_ifft_dzx.h5" %(grid_nodes,round(std_dev_phys,3)), 'r')
f6=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/hessian_comp/fil__DTFE_gd%d_smth%sMpc_ifft_dzy.h5" %(grid_nodes,round(std_dev_phys,3)), 'r')
ifft_dxx=f1['/hessian_ingr/ifft_dxx'][nrows_in:nrows_fn]
ifft_dxy=f2['/hessian_ingr/ifft_dxy'][nrows_in:nrows_fn]
ifft_dyy=f3['/hessian_ingr/ifft_dyy'][nrows_in:nrows_fn]
ifft_dzz=f4['/hessian_ingr/ifft_dzz'][nrows_in:nrows_fn]
ifft_dzx=f5['/hessian_ingr/ifft_dzx'][nrows_in:nrows_fn]
ifft_dzy=f6['/hessian_ingr/ifft_dzy'][nrows_in:nrows_fn]

#grid_nodes is translated depending upon the rows called for hessian compilation
grid_nodes_true=int(round(np.power((nrows_fn-nrows_in),1.*1/3)))
hessian=np.column_stack((ifft_dxx,ifft_dxy,ifft_dzx,ifft_dxy,ifft_dyy,ifft_dzy,ifft_dzx,ifft_dzy,ifft_dzz))
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
del ifft_dxx
del ifft_dxy
del ifft_dyy
del ifft_dzz
del ifft_dzx
del ifft_dzy
#hessian=np.split(hessian,grid_nodes_true**3)
hessian=np.reshape(hessian,(grid_nodes_true**3,3,3))#change to 3,3 for 3d and 1,1 for 1d
#calculate eigenvalues and eigenvectors
eig_vals_vecs=np.linalg.eig(hessian)
del hessian
#create unique randomness to each eigenvalue    
#np.random.seed(1)
#unq_rnd=np.round(np.random.rand(grid_nodes**3,3),3).astype(np.float32)
#unq_rnd=unq_rnd*10e-13
#extract eigenvalues
eigvals_unsorted=eig_vals_vecs[0]
eigvals=np.sort(eigvals_unsorted)
#extract eigenvectors
eigvecs=eig_vals_vecs[1]
eig_one=eigvals[:,2]
eig_two=eigvals[:,1]
eig_three=eigvals[:,0]
del eigvals

#link eigenvalues as keys to eigenvectors as values inside dictionary    
vec_arr_num,vec_row,vec_col=np.shape(eigvecs)
values=np.reshape(eigvecs.transpose(0,2,1),(vec_row*vec_arr_num,vec_col))#orient eigenvectors so that each row is an eigenvector
values=skl.normalize(values)
del eigvecs
eigvals_unsorted=eigvals_unsorted.flatten()


lss=['void','sheet','filament','cluster']#Choose which LSS you would like to get classified
def lss_classifier(lss,eigvals_unsorted,values,eig_one,eig_two,eig_three):
    
    ####Classifier#### 
    '''
    This is the classifier function which takes input:
    
    lss: the labels of Large scale structure which will be identified pixel by pixel and also eigenvectors 
    will be retrieved if applicable.
    
    vecsvals: These are the eigenvalues and eigevector pairs which correspond row by row.
    
    eig_one,two and three: These are the isolated eigenvalues 
    
    This function will output:
    
    eig_fnl: An array containing all of the relevent eigenvectors for each LSS type
    
    mask_fnl: array prescribing 0-void, 1-sheet, 2-filament and 3-cluster
    
    '''
    eig_fnl=np.zeros((grid_nodes_true**3,4))
    mask_fnl=np.zeros((grid_nodes_true**3))
    for i in lss:
        vecsvals=np.column_stack((eigvals_unsorted,values))
        recon_img=np.zeros([grid_nodes_true**3])
        if (i=='void'):
            recon_filt_one=np.where(eig_three>0)
            recon_filt_two=np.where(eig_two>0)
            recon_filt_three=np.where(eig_one>0)
        if (i=='sheet'):
            recon_filt_one=np.where(eig_three<0)
            recon_filt_two=np.where(eig_two>=0)
            recon_filt_three=np.where(eig_one>=0)
        if (i=='filament'):
            recon_filt_one=np.where(eig_three<0)
            recon_filt_two=np.where(eig_two<0)
            recon_filt_three=np.where(eig_one>=0)
        if (i=='cluster'):
            recon_filt_one=np.where(eig_three<0)
            recon_filt_two=np.where(eig_two<0)
            recon_filt_three=np.where(eig_one<0)
        
        #LSS FILTER#
        
        recon_img[recon_filt_one]=1
        recon_img[recon_filt_two]=recon_img[recon_filt_two]+1
        recon_img[recon_filt_three]=recon_img[recon_filt_three]+1  
        del recon_filt_one
        del recon_filt_two
        del recon_filt_three
        recon_img=recon_img.flatten()
        recon_img=recon_img.astype(np.int8)
        mask=(recon_img !=3)#Up to this point, a mask is created to identify where there are NO filaments...
        mask_true=(recon_img ==3)
        del recon_img
        vecsvals=np.reshape(vecsvals,(grid_nodes_true**3,3,4))
        
        
        #Find relevent eigpairs
        if (i=='void'):#There is no appropriate axis of a void?
            mask_fnl[mask_true]=0
            del mask_true
            
        if (i=='sheet'):
            vecsvals[mask,:,:]=np.ones((3,4))*9#...which are then converted into -9 at this point
            del mask
            fnd_prs=np.where(vecsvals[:,:,0]<0)#find LSS axis
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]
            mask_fnl[mask_true]=1
            del mask_true

        if (i=='filament'):
            vecsvals[mask,:,:]=np.ones((3,4))*-9#...which are then converted into -9 at this point
            del mask
            fnd_prs=np.where(vecsvals[:,:,0]>=0)#find LSS axis
            eig_fnl[fnd_prs[0],:]=vecsvals[fnd_prs[0],fnd_prs[1],:]
            mask_fnl[mask_true]=2
            del mask_true
            
        if (i=='cluster'):#There is no appropriate axis of a void?
            mask_fnl[mask_true]=3
            del mask_true            
        
    return eig_fnl,mask_fnl  
    
eig_fnl,mask_fnl= lss_classifier(lss,eigvals_unsorted,values,eig_one,eig_two,eig_three)#Function run
del eig_three
del eig_two
del eig_one  
del eigvals_unsorted
del values  
recon_vecs_x=eig_fnl[:,1]
recon_vecs_y=eig_fnl[:,2]
recon_vecs_z=eig_fnl[:,3]

f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d.h5" %(grid_nodes,round(std_dev_phys,3),part), 'w')
f.create_dataset('/group%d/x'%part,data=recon_vecs_x)
f.create_dataset('/group%d/y'%part,data=recon_vecs_y)
f.create_dataset('/group%d/z'%part,data=recon_vecs_z)
f.close()

f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d_mask.h5" %(grid_nodes,round(std_dev_phys,3),part), 'w')
f2.create_dataset('/mask%d'%part,data=mask_fnl)
f2.close()
