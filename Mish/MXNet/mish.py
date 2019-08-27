# MXNet Implementation of Mish Activation Function.

# Import Necessary Modules.
import mxnet as mx
import mxnet.ndarray as F

class Mish(mx.gluon.HybridBlock):
    def __init__(self):
        super(Mish, self).__init__()

    def hybrid_forward(self, x):        
        return x * F.tanh(F.activation(data = x, act_type = 'softrelu'))
