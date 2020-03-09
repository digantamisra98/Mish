'''
Script provides functional interface for Mish activation function.
Experimental device based functional definition. Not advised to use!
'''

# import pytorch
import torch
import torch.nn.functional as F


def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    if torch.cuda.is_available():
        return cuda_mish(input)
    else:
        return cpu_mish(input)

@torch.jit.script
def cuda_mish(input):
    return input * torch.tanh(F.softplus(input))

@torch.jit.script
def cpu_mish(input):
    delta = torch.exp(-input)
    alpha = 1 + 2 * delta
    return input * alpha / (alpha + 2* delta * delta)
