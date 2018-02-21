import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

sim_sz=250
grid_nodes=850
f1=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/my_den/den_grid%s_halo_bin_bolchoi" %grid_nodes, 'r')
a=f1['/halo'][:]
f1.close()
a=np.reshape(a,(grid_nodes,grid_nodes,grid_nodes))
smth_scales=np.array([1.12,2.8,3.92,5.6])#1,2,3.5,5Mpc/h smoothing scales
for smoothing in smth_scales:
    #smooth via convolution
    s=smoothing
    in_val,fnl_val=-140,140
    X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,grid_nodes),np.linspace(in_val,fnl_val,grid_nodes),np.linspace(in_val,fnl_val,grid_nodes))
    h=(1/np.sqrt(2*np.pi*s*s))**(3)*np.exp(-1/(2*s*s)*(Y**2+X**2+Z**2))
    
    h=np.roll(h,int(grid_nodes/2),axis=0)
    h=np.roll(h,int(grid_nodes/2),axis=1)
    h=np.roll(h,int(grid_nodes/2),axis=2)
    fft_dxx=np.fft.fftn(h)
    fft_db=np.fft.fftn(a)
    ifft_a=np.fft.ifftn(np.multiply(fft_dxx,fft_db)).real
    
    grid_phys=1.*sim_sz/grid_nodes#Size of each voxel in physical units
    val_phys=1.*(2*fnl_val)/grid_nodes#Value in each grid voxel
    std_dev_phys=1.*s/val_phys*grid_phys
    
    f=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/my_den/den_grid%s_halo_bin_bolchoi_smth%s" %(grid_nodes,std_dev_phys), 'w')
    f.create_dataset('/denssmth',data=ifft_a)
    f.close()
    
    #code to calculate pixel mass distribution. can be used for smoothed/non-smoothed density field
    ifft_a=ifft_a.flatten()
    bins=500
    data=np.histogram(ifft_a,bins=bins,density=True)
    bin_vals=np.delete(data[1],len(data[1])-1,0)
    plt.plot(np.log10(bin_vals),np.log10(data[0]),label='%sMpc/h smoothed'%round(std_dev_phys,2))


a=a.flatten()
bins=500
data=np.histogram(a,bins=bins,density=True)
bin_vals=np.delete(data[1],len(data[1])-1,0)
plt.plot(np.log10(bin_vals),np.log10(data[0]),label='raw')

plt.legend()
plt.xlabel('Log10(Mass)') 
plt.ylabel( 'Log10(p(mass))')    
plt.savefig('/scratch/GAMNSCM2/bolchoi_z0/investigation/mass_distribution_smthnnon_field_loglog_overplt_1_2_3.5_5Mpch.png')
