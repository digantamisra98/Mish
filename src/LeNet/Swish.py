## Import Necessary Module
import tensorflow as tf
import keras.backend as K

## For designing a custom tensorflow layer. Usage in ResNet-Module Building
def swish_layer(x):
  return tf.keras.layers.Lambda(lambda x: x*K.sigmoid(x))(x)

## For Mish activation function
def swish(x):
    return K.sigmoid(x)
