import numpy as np
import matplotlib.pyplot as plt
import math as mth
import h5py

f=h5py.File("/import/oth3/ajib0457/bolchoi_z0/investigation/my_denbox/dp_eigvecs-eigvecs_filament_two.h5" , 'r')
costheta=f['/dp'][:]
f.close()

costheta=abs(costheta)
c=np.linspace(-0.99,0.99,3000)#-1.5 to0.99 for the log-likelihood since (1-c). -1 to 1 for likelihood
loglike=np.zeros((len(c),1))
for i in range(len(c)):
        
    loglike[i,0]=np.sum(np.log((1-c[i])*np.sqrt(1+(c[i]/2))*(1-c[i]*(1-1.5*(costheta)**2))**(-1.5)))#log-likelihood

#Convert loglike into likelihood function (real space)
loglike_like=np.exp(loglike)
max_val_loglike=np.where(loglike==np.max(loglike))[0]#be careful that this condition avoids nan values within array
c_value_loglike=c[max_val_loglike]

print('MLV_loglike: %s'%c_value_loglike)
#error calculation at full width half maximum
FWHM_hf_hght=1.*(np.max(loglike_like)-np.min(loglike_like))/2
FWHM_index=abs(abs(loglike_like)-FWHM_hf_hght).argmin()
sig_c=c_value_loglike-c[FWHM_index]
print('error %s'%sig_c)
'''
#Create Gaussian to manually fit to likelihood function
mu=c_value_loglike
#variance=0.0058**2
sigma=0.00003
#Gaussian function
x=np.linspace(-1,1,1000)
normstdis=np.zeros((1000,1))
normstdis=1/(np.sqrt(2*(sigma**2)*mth.pi))*np.exp(-((x-mu)**2)/(2*sigma**2))
'''

fig=plt.figure()
plt.plot(c,np.exp(loglike),color='b',label='loglike using binned data')

plt.xlabel('Parameter (c)')
plt.ylabel('Likelihood')
plt.title('Likelihood Curve: c= %s and error=%s'%(c_value_loglike,sig_c))#how to make complete sigma number show up in title?
plt.legend()
#plt.savefig('/suphys/ajib0457/snapshot_012_LSS_class/correl/DTFE/plots/MLE_grid_methd/max-likelihood_ndotprodbins%s_c%s_error%s_grid%d_fil_Log%s-%s_smth%skpc_binr.png'%(n_dot_prod/2,c_value_loglike,sig_c,grid_nodes,low_int_mass,hi_int_mass,std_dev_phys))

#data_mod overplot
bins=15
#plotting data
data=np.histogram(costheta,bins=bins,density=True)
bin_vals=np.delete(data[1],len(data[1])-1,0)
figure=plt.figure(figsize=(8,5),dpi=50)
plt.plot(bin_vals,data[0],color='r',label='data')
plt.axhline(y=1, xmin=0, xmax=15, color = 'k',linestyle='--')

dotprodval=np.round(np.linspace(0,bin_vals[len(bin_vals)-1],1000),3)#value for each index in costheta array
#plotting model with errors
model=(1-c_value_loglike)*np.sqrt(1+1.*(c_value_loglike/2))*(1-c_value_loglike*(1-1.5*(dotprodval)**2))**(-1.5)
model_min=(1-(c_value_loglike-sig_c))*np.sqrt(1+((c_value_loglike-sig_c)/2))*(1-(c_value_loglike-sig_c)*(1-1.5*(dotprodval)**2))**(-1.5)
model_max=(1-(c_value_loglike+sig_c))*np.sqrt(1+((c_value_loglike+sig_c)/2))*(1-(c_value_loglike+sig_c)*(1-1.5*(dotprodval)**2))**(-1.5)

plt.plot(dotprodval,model,color='b',label='model')
plt.plot(dotprodval,model_min,color='g',label='model_error_min')
plt.plot(dotprodval,model_max,color='y',label='model_error_max')
#plt.axis((0,1,0.8,1.2))
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),ncol=3, fancybox=True, shadow=True)
plt.legend(loc='upper left')
