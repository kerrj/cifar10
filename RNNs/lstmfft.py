import keras
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as p
import scipy.io.wavfile as w
import fftunpack as f
import time 

def parse_data(data,look_back):
    x=np.array([data[0:look_back]])
    y=np.array([data[1]])    
    for i in range(look_back+1,len(data)-1):
        x=np.concatenate((x,np.array([data[i-look_back:i]])),0)
        y=np.concatenate((y,[data[i]]),0)
    return x,y

def step(data, model,batch_size):
    inp=np.zeros((batch_size,1,1))
    inp[0][0][0]=data[data.size-1:data.size]
    data=np.concatenate((data,model.predict(inp,batch_size=batch_size)[0]),0)
    return data

def compute_range(startpoint,npoints,model,batch_size):
    model.reset_states()    
    data=np.array([startpoint])
    for n in range(npoints):
        data=step(data,model,batch_size)
    return data

def compute_range_with_seed(data,start_index,seed_length,npoints,model,batch_size):
    model.reset_states()    
    data=data[start_index:start_index+seed_length]
    for n in data:
        model.predict(n.reshape((1,1,1)))
    
    for n in range(npoints):
        data=step(data,model,batch_size)
    return data

rate,data=w.read('/users/justin/Documents/tensorflow/bird.wav')
data=data[1000:30002]
p.show()
data,transform_size=f.unpack_data(data,rate,.01)
x,y=parse_data(data,1)
print('input shape',x.shape,'output shape',y.shape)

#for n in range(50):
    #p.plot(data[n],'.')
    #p.show()

batch_size=1

#model=keras.models.Sequential()
#model.add(keras.layers.LSTM(128,input_shape=(1,2*transform_size),stateful=True,batch_input_shape=(batch_size,1,2*transform_size),return_sequences=True))
#model.add(keras.layers.Dropout(.2))
#model.add(keras.layers.LSTM(128,stateful=True))
#model.add(keras.layers.Dropout(.2))
#model.add(keras.layers.Dense(2*transform_size))
#print(model.output)
#model.compile(keras.optimizers.Adam(),keras.losses.mean_absolute_error)


model=keras.models.load_model('/users/justin/Documents/tensorflow/models/lstm3.h5')
model.reset_states()

for n in range(20):
    model.fit(x,y,epochs=1,batch_size=batch_size,shuffle=False)
    model.reset_states()
model.save('/users/justin/Documents/tensorflow/models/lstm3.h5')

prediction=model.predict(x,batch_size=1)
for n in range(1):
    p.plot(f.unpack_fft(prediction))
    p.show()

##LSTM1 sin 
##lstm2 qudratic
##lstm3 wav file