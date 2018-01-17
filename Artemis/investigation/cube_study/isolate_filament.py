import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl
grid_nodes=100
np.random.seed(1)
a=np.random.rand(grid_nodes**3,3)
a_norm=skl.normalize(a)
a_x=np.reshape(a_norm[:,0],(grid_nodes,grid_nodes,grid_nodes))
a_y=np.reshape(a_norm[:,1],(grid_nodes,grid_nodes,grid_nodes))
a_z=np.reshape(a_norm[:,2],(grid_nodes,grid_nodes,grid_nodes))
#plt.quiver(a_x[0,:,:], a_y[0,:,:], a_z[0,:,:], alpha=.5)

#slices & Circle centre initial & radius details
slc_min,slc_max=15,30
l,m,r=0,50,90#Its if the indices are within the radius, then I commit a 1 to the mask... so I will need 

x,y,z=np.shape(a_x)
a_x_new=np.zeros((x,y,z))
a_y_new=np.zeros((x,y,z))
a_z_new=np.zeros((x,y,z))
for i in range(slc_max-slc_min):
    l=slc_min+i#coordinate tracking across slices
    l=2*l#Function with which to guide circle centre
    for j in range(y):
        for k in range(z):  
            if ((j-l)**2+(k-m)**2<=r):
                a_x_new[i+slc_min,j,k]=a_x[i+slc_min,j,k]
                a_y_new[i+slc_min,j,k]=a_y[i+slc_min,j,k]
                a_z_new[i+slc_min,j,k]=a_z[i+slc_min,j,k]

#Plot all slices involved
fig, ax = plt.subplots(figsize=(5,30),dpi=100)
for slc in range(slc_max-slc_min):
    ax=plt.subplot2grid((slc_max-slc_min,1), (slc,0))
    slc=slc_min+slc-1
    vecs=np.column_stack((a_y_new[slc,:,:].flatten(),a_z_new[slc,:,:].flatten()))#flatten and stack
    vec_norm=skl.normalize(vecs)#normalize
    a_y_new_norm=np.reshape(vec_norm[:,0],(grid_nodes,grid_nodes))#reshape
    a_z_new_norm=np.reshape(vec_norm[:,1],(grid_nodes,grid_nodes))
    
    plt.quiver(a_y_new_norm, a_z_new_norm, alpha=.5,scale=50)
plt.savefig('circle_slices.png')
