<p align="center">
  <img width="300" src="Observations/logo_transparent.png">
</p>

[![Donate](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![HitCount](http://hits.dwyl.io/digantamisra98/Mish.svg)](http://hits.dwyl.io/digantamisra98/Mish)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/digantamisra98/Mish/issues)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/DigantaMisra?locale.x=en_GB)
[![codecov](https://codecov.io/gh/digantamisra98/Mish/branch/master/graph/badge.svg?token=wrRQ8RM3R1)](https://codecov.io/gh/digantamisra98/Mish)
[![Known Vulnerabilities](https://snyk.io/test/github/digantamisra98/Mish/badge.svg)](https://snyk.io/test/github/digantamisra98/Mish)


# Mish: Self Regularized Non-Monotonic Activation Function

Inspired by *Swish* Activation Function ([Paper](https://arxiv.org/abs/1710.05941)), **Mish** is a Self Regularized Non-Monotonic Neural Activation Function. Activation Function serves a core functionality in the training process of a Neural Network Architecture and is represented by the basic mathematical representation: 
<div style="text-align:center"><img src ="Observations/act.png"  width="500"/></div>
<em> Image Credits: https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Activation_Functions
</em><br>
<br>
An Activation Function is generally used to introduce non-linearity and over the years of theoretical machine learning research, many activation functions have been constructed with the 2 most popular amongst them being: 

- ReLU (Rectified Linear Unit; f(x)=max(0,x)) <br>
- TanH <br>

Other notable ones being: <br> 
- Softmax (Used for Multi-class Classification in the output layer) <br> 
- Sigmoid (f(x)=(1+e<sup>-x</sup>)<sup>-1</sup>;Used for Binary Classification and Logistic Regression) <br>
- Leaky ReLU (f(x)=0.001x (x<0) or x (x>0)) <br>



## Dependencies:
- TensorFlow = 1.12.x or higher
- Keras = 2.2.x or higher
- Python = 3x

## Mathematics under the hood:

Mish Activation Function can be mathematically represented by the following formula:<br> 
<div style="text-align:center"><img src ="Observations/imgtemp_ugysxo-1.png"  width="220"/></div>
It can also be represented by using the SoftPlus Activation Function as shown:<br><br>
<div style="text-align:center"><img src ="Observations/imgtemp_x5rglu-1.png"  width="185"/></div>
<div style="text-align:center"><img src ="Observations/imgtemp_utahjs-1.png"  width="320"/></div><br>
And it's 1<sup>st</sup> and 2<sup>nd</sup> derivatives are given below:<br>
<div style="text-align:center"><img src ="Observations/d1.png"  width="185"/></div>
<div style="text-align:center"><img src ="Observations/imgtemp_qph7sj-1.png"  width="320"/></div><br>
Where:<br>
<div style="text-align:center"><img src ="Observations/delta.png"  width="200"/></div>
<div style="text-align:center"><img src ="Observations/imgtemp_3rbfba-1.png"  width="270"/></div>
<br>
When visualized, Mish Activation Function closely resembles the function path of Swish having a small decay (preserve) in the negative side while being near linear on the positive side. It is a Monotonic Function and as observed from it's derivatives functions shown above and graph shown below, it can be noted that it has a Non-Monotonic 1<sup>st</sup> derivative and 2<sup>nd</sup> derivative. <br>

**Mish** ranges between ≈-0.31 to ∞.<br>
<div style="text-align:center"><img src ="Observations/Mish3.png"  width="800"/></div>
<div style="text-align:center"><img src ="Observations/Derivatives.png"  width="800"/></div>
Based on mathematical analysis, it is also confirmed that the function has a parametric order of continuity of:<div style="text-align:center"><img src ="Observations/imgtemp_kyk9k1-1.png"  width="30"/></div>

**Mish** has a very sharp global minima similar to Swish, which might account to gradients updates of the model being stuck in the region of sharp decay thus may lead to bad performance levels as compared to ReLU. Mish, also being mathematically heavy, is more computationally expensive as compared to the time complexity of Swish Activation Function. 

## Set-Up:

All Experiments were performed on [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true) with NVIDIA Tesla T4 GPU. Initial Results were ran on a local machine equipped with an 64-bit, Intel i7-8750H CPU clocked at 2.2GHz, 16GB RAM with a NVIDIA GTX-1060 having the following specifications: 

<table>
  <tr><th colspan=2>GPU Specification Sheet</th></tr>
  <tr><td>Driver Version</td><td>418.81</td></tr>
  <tr><td>Driver Type</td><td>Standard</td></tr>
  <tr><td>CUDA Cores</td><td>1280</td></tr>
  <tr><td>Graphics Clock</td><td>1455 MHz</td></tr>
  <tr><td>Memory Data Rate</td><td>8.01 Gbps</td></tr>
  <tr><td>Memory Interface</td><td>192-bit</td></tr>
  <tr><td>Memory Bandwidth</td><td>192.19 GB/s</td></tr>
  <tr><td>Available Graphic Memory</td><td>14258 MB</td></tr>
</table>

## Try It! 

### Install Dependencies

```pip install -r requirements.txt```

- For LeNet with Mish on MNIST Google Colab Testing: 

> Run the [LeNet_Mish.ipynb](https://github.com/digantamisra98/Mish/blob/master/Notebooks/LeNet_Mish.ipynb) file on your Google Colaboratory

## Conclusion:

## Future Work (Coming Soon):

## Support Me

## Contact: 
- [LinkedIn](https://www.linkedin.com/in/misradiganta/)<br>
- Email: mishradiganta91@gmail.com
