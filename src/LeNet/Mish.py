## Import Necessary Module
import tensorflow as tf
import numpy as np

## For designing a custom tensorflow layer. Usage in ResNet-Module Building
def mish_layer(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.log(1+tf.exp(x))))(x)

## For Mish activation function
def mish(x):
    return tf.tanh(tf.log(1+tf.exp(x)))

### For Numpy version of Mish
def mish_np(x):
    return np.tanh(np.log(1+np.exp(x)))
