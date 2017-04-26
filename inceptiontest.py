import numpy as np
import matplotlib
import cv2
matplotlib.use('TkAgg')
import matplotlib.pyplot as p
import keras

net=keras.applications.InceptionV3();
img=cv2.imread('/users/justin/Documents/tensorflow/images/test.jpg')
net.predict(img);