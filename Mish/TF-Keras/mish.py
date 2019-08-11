"""Tensorflow-Keras Implementation of Mishh"""

## Import Necessary Modules
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Mish(Layer):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)
    '''

    def __init__(self):
        super(Mish, self).__init__()
    
    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.log(1 + tf.math.exp(inputs)))