import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py


grid_nodes=850
f1=h5py.File("/scratch/GAMNSCM2/bolchoi_z0/my_den/den_grid%s_halo_bin_bolchoi" %grid_nodes, 'r')
a=f1['/halo'][:]
f1.close()
a=np.reshape(a,(grid_nodes,grid_nodes,grid_nodes))
#smooth via convolution
s=3.92
in_val,fnl_val=-140,140
X,Y,Z=np.meshgrid(np.linspace(in_val,fnl_val,grid_nodes),np.linspace(in_val,fnl_val,grid_nodes),np.linspace(in_val,fnl_val,grid_nodes))
h=(1/np.sqrt(2*np.pi*s*s))**(3)*np.exp(-1/(2*s*s)*(Y**2+X**2+Z**2))

h=np.roll(h,int(grid_nodes/2),axis=0)
h=np.roll(h,int(grid_nodes/2),axis=1)
h=np.roll(h,int(grid_nodes/2),axis=2)
fft_dxx=np.fft.fftn(h)
fft_db=np.fft.fftn(a)
ifft_a=np.fft.ifftn(np.multiply(fft_dxx,fft_db)).real



#code to calculate pixel mass distribution. can be used for smoothed/non-smoothed density field
ifft_a=ifft_a.flatten()
bins=100
data=np.histogram(ifft_a,bins=bins,density=True)
bin_vals=np.delete(data[1],len(data[1])-1,0)
plt.plot(np.log10(bin_vals),data[0])

plt.xlabel('Mass') 
plt.ylabel( 'p(mass)')    
plt.savefig('/scratch/GAMNSCM2/bolchoi_z0/investigation/mass_distribution_smoothed_field_semilogx.png')
