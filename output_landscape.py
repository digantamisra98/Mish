import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
​
from collections import OrderedDict
import numpy as np
​
from PIL import Image
​
from sklearn.preprocessing import MinMaxScaler
​
# import matplotlib for visualization
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
​
def fswish(input, beta=1.25):
    return input * torch.sigmoid(beta * input)
​
class swish(nn.Module):
    def __init__(self, beta = 1.25):
        '''
        Init method.
        '''
        super().__init__()
        self.beta = beta
​
​
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return fswish(input, self.beta)
​
def fmish(input):
    return input * torch.tanh(F.softplus(input))
​
class mish(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()
​
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return fmish(input)
​
def build_model(activation_function):
    return nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2, 64)),
                          ('activation1', activation_function), # use custom activation function
                          ('fc2', nn.Linear(64, 32)),
                          ('activation2', activation_function),
                          ('fc3', nn.Linear(32, 16)),
                          ('activation3', activation_function),
                          ('fc4', nn.Linear(16, 1)),
                          ('activation4', activation_function)]))
​
def convert_to_PIL(img, width, height):
    img_r = img.reshape(height,width)
​
    pil_img = Image.new('RGB', (height,width), 'white')
    pixels = pil_img.load()
​
    for i in range(0, height):
        for j in range(0, width):
            pixels[j, i] = img[i,j], img[i,j], img[i,j]
​
    return pil_img
​
def main():
    model1 = build_model(nn.ReLU())
    model2 = build_model(swish(beta = 1))
    model3 = build_model(mish())
​
    x = np.linspace(0.0, 10.0, num=100)
    y = np.linspace(0.0, 10.0, num=100)
​
    grid = [torch.tensor([xi, yi]) for xi in x for yi in y]
​
    np_img_relu = np.array([model1(point).detach().numpy() for point in grid]).reshape(100, 100)
    np_img_swish = np.array([model2(point).detach().numpy() for point in grid]).reshape(100, 100)
    np_img_mish = np.array([model3(point).detach().numpy() for point in grid]).reshape(100, 100)
​
    scaler = MinMaxScaler(feature_range=(0, 255))
    np_img_relu = scaler.fit_transform(np_img_relu)
    np_img_swish = scaler.fit_transform(np_img_swish)
    np_img_mish = scaler.fit_transform(np_img_mish)
​
    image_relu = convert_to_PIL(np_img_relu, 100, 100)
    image_swish = convert_to_PIL(np_img_swish, 100, 100)
    image_mish = convert_to_PIL(np_img_mish, 100, 100)
​
    image_relu.save('relu.png')
    image_swish.save('swish.png')
    image_mish.save('mish.png')
​
    plt.imsave('imrelu.png', np_img_relu)
    plt.imsave('imswish.png', np_img_swish)
    plt.imsave('imsmish.png', np_img_mish)
​
    return
​
​
if __name__ == '__main__':
    main()
