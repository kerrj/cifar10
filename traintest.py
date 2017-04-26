import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as p
import keras

activation_function=keras.activations.sigmoid
model=keras.models.Sequential()
model.add(keras.layers.Dense(10,activation=activation_function,input_dim=2))
model.add(keras.layers.Dense(1,activation=activation_function))
model.compile(optimizer=keras.optimizers.SGD(lr=1),loss=keras.losses.mean_squared_error)

#plot results
iterations=10000
history=model.fit(np.array([[1,0],[0,1],[1,1],[0,0]]),np.array([1,1,0,0]),epochs=iterations,verbose=1,batch_size=10)
p.semilogy(list(range(iterations)),history.history.get('loss'))
p.title('Loss vs Iteration')
print('results',model.predict(np.array([[1,0],[0,1],[1,1],[0,0]])))
p.show()
