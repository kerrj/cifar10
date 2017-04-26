import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pylab

sess=tf.Session()

#making a linear fit model
a=tf.Variable([.1])
b=tf.Variable([-.1])
c=tf.Variable([.2])
x=tf.placeholder(tf.float32)
model=a*tf.square(x)+b*x+c

y=tf.placeholder(tf.float32)
difference=tf.square(y-model);
loss=tf.reduce_sum(difference);

optimizer=tf.train.GradientDescentOptimizer(.0001)
train=optimizer.minimize(loss)

#initialize our variables
init=tf.global_variables_initializer()
sess.run(init)
xdata=[0,1,2,3,4,5,6]
ydata=[0,1,4,9,16,25,36]

for n in range(2000):
    sess.run(train,{x:xdata,y:ydata})
print(sess.run([a,b,c]))
pylab.plot(xdata,ydata,'b.')
plotx=np.arange(0,5,.1)
ploty=np.square(plotx)*sess.run(a)+sess.run(b)*plotx+sess.run(c)
pylab.plot(plotx,ploty,'r-')
pylab.show()