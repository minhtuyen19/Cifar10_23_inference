from tensorflow import keras
from keras.datasets import cifar10
from keras import datasets
from keras.utils import np_utils
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model

model = keras.models.load_model('cifar10_newkeras.h5')
# model.summary()
 
# weights = model.layers[1].get_weights()[0]
# print(weights.reshape(32, 32, 3, 3))
# print(weights.shape)

with open('D:/ASSET/NAM4HK2/Predict/Readmodel/Read_cifar10_newkeras/bias_10.txt', 'w') as f:
        weights = model.layers[22].get_weights()
        #layer_name = model.layers[i].name
        # f.write(layer_name + '\n')
        weights = np.array(weights[1])
        # weights = np.array((weights[0].reshape(10,128))).flatten()#.reshape(10, 128))
        # weights = weights.flatten()[4609]
        # weights = np.transpose(weights, (1,0)).flatten()
        print(weights.shape)
        np.set_printoptions(threshold=np.inf, linewidth=200)
        # # # print(weights)
        f.write(str(weights) + '\n')