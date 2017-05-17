import tensorflow.contrib.keras as keras
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as p
import numpy as np
import cifar10test

def get_label(img,model,labels,answer):
    result=model.predict(img);
    maxvalue=np.max(result)
    for i in range(10):
        if result[0,i]==maxvalue:
            labelresult=labels[i]
            for j in range(10):
                if answer[0,j]==1:
                    correctresult=labels[j]
            return 'Net answer: '+str(labelresult)+', Confidence:'+str(result[0,i])+', Correct answer: '+str(correctresult)
                
            
         
#change this for testing different models
modelname='cifar.h5'
model=keras.models.load_model('/users/justin/Documents/tensorflow/models/'+modelname)
testlabels=np.load('/users/justin/Documents/tensorflow/images/cifar/testlabels.npy')
testimgs=np.load('/users/justin/Documents/tensorflow/images/cifar/test.npy')

d=cifar10test.unpickle('/users/justin/Documents/tensorflow/cifar-10/batches.meta')
labels=d[b'label_names']

i=1000;
while True:
    p.imshow(testimgs[i])
    p.title(get_label(testimgs[i:i+1],model,labels,testlabels[i:i+1]))
    p.show(block=False)
    p.waitforbuttonpress(timeout=-1)
    i=i+1;