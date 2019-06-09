'''
Script provides functional interface for custom activation functions.
'''

# import pytorch
import torch
from torch import nn
import torch.nn.functional as F

def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))
