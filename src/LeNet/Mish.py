## Import Necessary Module
import tensorflow as tf
import numpy as np
from keras import backend as K

## For designing a custom tensorflow lambda layer. Usage in ResNet-Module Building
def mish_layer(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.softplus(x)))(x)

## For designing a custom keras lambda layer. Usage in ResNet-Module Building
def mish_layer_keras(x):
    return keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)

## For Mish activation function in tensorflow
def mish_tf(x):
    return x*tf.tanh(tf.math.softplus(x))

## For Mish activation function in Keras
def mish_K(x):
    return x*K.tanh(K.softplus(x))

### For Numpy version of Mish
def mish_np(x):
    return np.tanh(np.log(1+np.exp(x)))
