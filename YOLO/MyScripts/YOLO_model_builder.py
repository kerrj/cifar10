import keras
import keras.backend as B
print(B.image_dim_ordering(),B.image_data_format())

def build_model():
    model=keras.models.Sequential()
    model.add(keras.layers.Convolution2D(64,(7,7),strides=(2,2),input_shape=(448,448,3),activation='relu',padding='same'))
    model.add(keras.layers.MaxPool2D(strides=(2,2)))
    
    model.add(keras.layers.Convolution2D(192,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.MaxPool2D(strides=(2,2)))
    
    model.add(keras.layers.Convolution2D(128,kernel_size=(1,1),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(256,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(256,kernel_size=(1,1),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(512,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.MaxPool2D(strides=(2,2)))
    
    #x4
    model.add(keras.layers.Convolution2D(256,kernel_size=(1,1),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(512,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(256,kernel_size=(1,1),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(512,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(256,kernel_size=(1,1),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(512,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(256,kernel_size=(1,1),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(512,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(512,kernel_size=(1,1),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(1024,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.MaxPool2D(strides=(2,2)))
    
    model.add(keras.layers.Convolution2D(512,kernel_size=(1,1),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(1024,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(512,kernel_size=(1,1),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(1024,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(1024,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(1024,kernel_size=(3,3),activation='relu',padding='same',strides=(2,2)))
    model.add(keras.layers.Convolution2D(1024,kernel_size=(3,3),activation='relu',padding='same'))
    model.add(keras.layers.Convolution2D(1024,kernel_size=(3,3),activation='relu',padding='same'))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096,activation='relu'))
    model.add(keras.layers.Dense(490,activation='relu'))
    model.add(keras.layers.Reshape(target_shape=(7,7,10)))
    model.compile(loss=keras.losses.binary_crossentropy,optimizer='adadelta')
    return model

def loss_function():
    return
    
m=build_model()
print(m.outputs)