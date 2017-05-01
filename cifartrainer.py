import numpy as np
import keras
import cifar10test
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as p

iterations=20

#change this to train different models
modelname='cifar.h5'
#load training data
traininglabels=np.load('/users/justin/Documents/NeuralNets/images/cifar/labels.npy')
trainingimgs=np.load('/users/justin/Documents/NeuralNets/images/cifar/images.npy')

#load test data
testlabels=np.load('/users/justin/Documents/NeuralNets/images/cifar/testlabels.npy')
testimgs=np.load('/users/justin/Documents/NeuralNets/images/cifar/test.npy')

#load model
model=keras.models.load_model('/users/justin/Documents/NeuralNets/models/'+modelname)
#model=cifar10test.compile_model()


#model.optimizer=keras.optimizers.Adadelta(lr=1)
print(model.optimizer.get_config())
#fit model
history=model.fit(trainingimgs,traininglabels,epochs=iterations,batch_size=1000)



#evaluate model with test set
print(model.evaluate(testimgs,testlabels))

#resave model, this overwrites the other model
model.save('/users/justin/Documents/NeuralNets/models/'+modelname)


p.plot(history.history['loss'])
p.show()
#cifar.h5 has dropout of .2 between everything using adadelta, accuracy of about .72
#cifar2.h5 has dropout of .4 between everything using adadelta, accuracy of about .62