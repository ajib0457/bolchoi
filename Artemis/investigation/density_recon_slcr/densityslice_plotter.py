import numpy as np
import h5py
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

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

mask=np.zeros((grid_nodes**3))
for part in range(tot_parts):#here I have to figure out how these have been stored then put them back together, probably just column stack all
#into 1 array
    nrows_in=int(1.*(grid_nodes**3)/tot_parts*part)
    nrows_fn=nrows_in+int(1.*(grid_nodes**3)/tot_parts)
    f2=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/correl/my_den/files/output_files/eigvecs/fil_recon_vecs_DTFE_gd%d_smth%sMpc_%d_mask.h5" %(grid_nodes,round(std_dev_phys,3),part), 'r')
    mask[nrows_in:nrows_fn]=f2['/mask%d'%part][:]
    f2.close()
mask=np.reshape(mask,(grid_nodes,grid_nodes,grid_nodes))#density field
f1=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/my_den/den_grid%s_halo_bin_bolchoi" %grid_nodes, 'r')
a=f1['/halo'][:]

#smooth density field
c=np.reshape(a,(grid_nodes,grid_nodes,grid_nodes))#density field
s=3.92#standard deviation
c_smoothed=ndimage.filters.gaussian_filter(c,s)

#slices options
slc=300

plt.figure(figsize=(20,20),dpi=100)
#The two function below are purely for the color scheme of the imshow plot: Classifier, used to create discrete imshow
def colorbar_index(ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))

def cmap_discretize(cmap, N):   
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in xrange(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

#Classifier: This subplot must be first so that the two functions above will help to discretise the color scheme and color bar
ax=plt.subplot2grid((2,1), (0,0))  
plt.title('Classifier')
#plt.xlabel('z')
#plt.ylabel('x')
cmap = plt.get_cmap('jet')#This is where you can change the color scheme
ax.imshow(np.rot90(mask[slc,:,:],1), interpolation='nearest', cmap=cmap,extent=[0,850,0,850])#The colorbar will adapt to data
colorbar_index(ncolors=4, cmap=cmap)

#Density field
ax5=plt.subplot2grid((2,1), (1,0))    
plt.title('classified image')
cmapp = plt.get_cmap('jet')
scl_plt=35#reduce scale of density fields and eigenvalue subplots by increasing number
dn_fl_plt=ax5.imshow(np.power(np.rot90(c_smoothed[slc,:,:],1),1./scl_plt),cmap=cmapp,extent=[0,850,0,850])#The colorbar will adapt to data
plt.colorbar(dn_fl_plt,cmap=cmapp)

plt.savefig('/scratch/GAMNSCM2/bolchoi_z0/investigation/bolchoi_recon_smthden_gd%d_slc%d_smth%sMpc_xplane.png' %(grid_nodes,slc,std_dev_phys))

