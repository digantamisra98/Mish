# MXNet Implementation of Mish Activation Function.

# Import Necessary Modules.
import mxnet as mx
import mxnet.ndarray as F

class Mish(mx.gluon.HybridBlock):
'''
    Mish Activation Function.

    .. math::

        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

'''
    def __init__(self):
        super(Mish, self).__init__()

    def hybrid_forward(self, x):        
        return x * F.tanh(F.activation(data = x, act_type = 'softrelu'))
