import keras
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as p

def parse_data(data,domain,look_back):
    x=np.array([[domain[0:look_back]]])
    y=np.array([data[0]])
    for i in range(look_back,len(data)):
        x=np.concatenate((x,np.array([[domain[i-look_back:i]]])),0)
        y=np.concatenate((y,[data[i]]),0)
    return x,y


look_back=1
batch_size=5
#model=keras.models.Sequential()
#model.add(keras.layers.LSTM(8,dropout=.5,recurrent_dropout=.1,input_shape=(1,look_back),stateful=True,batch_input_shape=(batch_size,1,look_back),return_sequences=True))
#model.add(keras.layers.LSTM(1,dropout=.5,recurrent_dropout=.1,input_shape=(1,look_back),stateful=True))
##model.add(keras.layers.Dense(1))
#model.compile(keras.optimizers.Adam(),keras.losses.mean_squared_error)

model=keras.models.load_model('/users/justin/Documents/tensorflow/models/lstm2.h5')

print(model.output)
domain=np.linspace(0,10*np.pi,1000)
data=np.sin(domain)
x,y=parse_data(data,domain,look_back)

model.fit(x,y,epochs=100,batch_size=batch_size)
model.save('/users/justin/Documents/tensorflow/models/lstm2.h5')
testdomain=np.linspace(0*np.pi,20*np.pi,2000)
testdata=np.sin(testdomain)
testx,testy=parse_data(testdata,testdomain,look_back)
p.plot(testdomain,model.predict(testx,batch_size=batch_size),testdomain,testdata)
p.show()

#lstm2, dropout .5, recurrentdropout .1
#lstm1, .2 dropout