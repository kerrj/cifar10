import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as p
import keras
import pickle

def unpickle(filename):
    with open(filename, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

iterations=5
def parse_labels(labels,num_labels):
    for n in range(num_labels):
        l=keras.utils.to_categorical(labels[n],num_classes=10)
        label=l[0]
        if n==0:
            labellist=[label]
        else:
            labellist.append(label)
    labelarray=np.array(labellist)
    return labelarray

def compile_model():
    #initialize our sequential CNN model
    model=keras.models.Sequential()
    model.add(keras.layers.Convolution2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),activation='relu'))
    model.add(keras.layers.Dropout(.4))
    model.add(keras.layers.pooling.MaxPooling2D(strides=(2,2)))    
    model.add(keras.layers.Convolution2D(filters=32,kernel_size=(3,3),activation='relu'))
    model.add(keras.layers.Dropout(.4))
    model.add(keras.layers.Convolution2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(keras.layers.pooling.MaxPooling2D(strides=(2,2)))    
    model.add(keras.layers.Dropout(.4))    
    model.add(keras.layers.Convolution2D(filters=64,kernel_size=(3,3),activation='relu'))
    model.add(keras.layers.Dropout(.4))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(.4))
    model.add(keras.layers.Dense(256,activation='relu'))
    model.add(keras.layers.Dropout(.4))
    model.add(keras.layers.Dense(10,activation='softmax'))
    model.compile(keras.optimizers.Adadelta(lr=1),loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
    print(model.outputs)
    
    return model