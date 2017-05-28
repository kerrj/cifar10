import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as p

def unpack_data(data,sample_rate,time_window):
    window_size=int(time_window*sample_rate)
    n_transforms=int(np.floor(data.size/window_size))
    transform_length=int(np.floor(window_size))
    output=np.zeros((n_transforms,2*transform_length))
    
    for n in range(n_transforms):
        index=int(window_size*n)
        fourier=np.fft.fft(data[index:index+window_size])
        output[n,0:transform_length]=np.real(fourier[0:transform_length])
        output[n,transform_length:]=np.imag(fourier[0:transform_length])
    return output,transform_length

def unpack_fft(fft_data):
    transform_length=int(fft_data.shape[1]/2)
    output=np.array([0])    
    for n in range(fft_data.shape[0]):
        fourier_data=fft_data[n,:]
        full_fourier=fourier_data[0:transform_length]+1j*fourier_data[transform_length:]
        data=np.fft.ifft(full_fourier)
        output=np.concatenate((output,data))
    return np.delete(output,0)
    
    
if(__name__=='__main__'):
    domain=np.linspace(0,100*np.pi,10000)
    r=np.cos(domain)+np.sin(2*domain)
    
    fourier,size=unpack_data(r,10000,.1)
    

    unpacked_data=unpack_fft(fourier)
    p.plot(unpacked_data,'.',r,'-')
    p.show()