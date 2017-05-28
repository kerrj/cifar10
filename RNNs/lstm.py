import keras
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as p
import scipy.io.wavfile as w

def parse_data(data,look_back):
    x=np.array([[data[0:look_back]]])
    y=np.array([data[1]])    
    for i in range(look_back+1,len(data)-1):
        x=np.concatenate((x,np.array([[data[i-look_back:i]]])),0)
        y=np.concatenate((y,[data[i]]),0)
    return x,y

def func(domain):
    return np.sin(domain)

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


look_back=1
domain_size=100
batch_size=1

##########Create model, either new or loaded
model=keras.models.Sequential()
model.add(keras.layers.LSTM(128,input_shape=(1,look_back),stateful=True,batch_input_shape=(batch_size,1,look_back),return_sequences=True))
model.add(keras.layers.Dropout(.4))
model.add(keras.layers.LSTM(128,stateful=True))
model.add(keras.layers.Dropout(.4))
model.add(keras.layers.Dense(1))
model.compile(keras.optimizers.Adam(),keras.losses.mean_absolute_error)

#model=keras.models.load_model('/users/justin/Documents/tensorflow/models/lstm3.h5')

################train model
#domain=np.linspace(0,2*np.pi,domain_size)
#data=func(domain)
#x,y=parse_data(data,look_back)


rate,data=w.read('/users/justin/Documents/tensorflow/bird.wav')
data=data[1000:10002]
min_data = np.min(data)
max_data = np.max(data)
data=(data - min_data) / (max_data - min_data)
x,y=parse_data(data,look_back)
print(x.shape)
model.reset_states()

for n in range(10):
    model.fit(x,y,epochs=1,batch_size=batch_size,shuffle=False)
    model.reset_states()
model.save('/users/justin/Documents/tensorflow/models/lstm3.h5')


###################test on training data
#testdomain=np.linspace(0,4*np.pi,2*domain_size)
#testdata=func(testdomain)
#testx,testy=parse_data(testdata,look_back)
#p.plot(testdomain[1:2*domain_size-look_back],model.predict(testx,batch_size=batch_size),testdomain,testdata)
#p.show()

d=compute_range_with_seed(data,0,100,1000,model,batch_size)
w.write('/users/justin/Documents/tensorflow/output.wav',rate,d)
p.plot(d)
p.title('Generated Range')
p.show()

model.reset_states()
p.plot(list(range(x.size)),model.predict(x,batch_size=batch_size),list(range(y.size)),y)
p.title('Range from training set')
p.show()
#LSTM1 sin 
#lstm2 qudratic
#lstm3 wav file