"""Tensorflow-Keras Implementation of Mish"""

## Import Necessary Modules
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils.generic_utils import get_custom_objects

class Mish(Activation):
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
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})
