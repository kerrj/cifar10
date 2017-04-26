import numpy as np
import pickle
from keras.utils import to_categorical

def unpickle(filename):
    with open(filename, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

def parse_labels(labels,num_labels):
    for n in range(num_labels):
        l=to_categorical(labels[n],num_classes=10)
        label=l[0]
        if n==0:
            labellist=[label]
        else:
            labellist.append(label)    
    labelarray=np.array(labellist)
    return labelarray

#d2=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_1')
#numpics=10000
#labels=d2[b'labels']
#masterlabelarray=parse_labels(labels,numpics)
#print(masterlabelarray.shape)
#d2=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_2')
#labels=d2[b'labels']
#labelarray=parse_labels(labels,numpics)
#masterlabelarray=np.append(masterlabelarray,labelarray,0)
#print(masterlabelarray.shape)
#d2=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_3')
#labels=d2[b'labels']
#labelarray=parse_labels(labels,numpics)
#masterlabelarray=np.append(masterlabelarray,labelarray,0)
#print(masterlabelarray.shape)
#d2=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_4')
#labels=d2[b'labels']
#labelarray=parse_labels(labels,numpics)
#masterlabelarray=np.append(masterlabelarray,labelarray,0)
#print(masterlabelarray.shape)
#d2=unpickle('/users/justin/Documents/tensorflow/cifar-10/data_batch_5')
#labels=d2[b'labels']
#labelarray=parse_labels(labels,numpics)
#masterlabelarray=np.append(masterlabelarray,labelarray,0)
#print(masterlabelarray.shape)
#np.save('/users/justin/Documents/tensorflow/images/cifar/labels.npy',masterlabelarray)

d2=unpickle('/users/justin/Documents/tensorflow/cifar-10/test_batch')
labels=d2[b'labels']
labelarray=parse_labels(labels,10000)
np.save('/users/justin/Documents/tensorflow/images/cifar/testlabels.npy',labelarray)