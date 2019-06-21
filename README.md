<p align="center">
  <img width="300" src="Observations/Activation Function.png">
</p>

[![Donate](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![HitCount](http://hits.dwyl.io/digantamisra98/Mish.svg)](http://hits.dwyl.io/digantamisra98/Mish)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/digantamisra98/Mish/issues)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/DigantaMisra?locale.x=en_GB)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f67cf4cf73bf47fbbe4b3dd6f78bb10b)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=digantamisra98/Mish&amp;utm_campaign=Badge_Grade)
[![CircleCI](https://circleci.com/gh/digantamisra98/Mish.svg?style=svg&circle-token=06a25b6387b645a32713b0ac47878adac8e52c3a)](https://circleci.com/gh/digantamisra98/Mish)

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
<div style="text-align:center"><img src ="Observations/imgtemp_ugysxo-1.png"  width="250"/></div>
It can also be represented by using the SoftPlus Activation Function as shown:<br><br>
<div style="text-align:center"><img src ="Observations/imgtemp_x5rglu-1.png"  width="200"/></div>
<div style="text-align:center"><img src ="Observations/imgtemp_utahjs-1.png"  width="340"/></div><br>
And it's 1<sup>st</sup> and 2<sup>nd</sup> derivatives are given below:<br>
<div style="text-align:center"><img src ="Observations/d1.png"  width="120"/></div>
<div style="text-align:center"><img src ="Observations/d2.png"  width="700"/></div><br>
Where:<br>
<div style="text-align:center"><img src ="Observations/delta.png"  width="200"/></div>
<div style="text-align:center"><img src ="Observations/omega.png"  width="400"/></div>
<br>

The Taylor Series Expansion of *f(x)* at *x=0* is given by: <br>
<div style="text-align:center"><img src ="Observations/series.png"  width="700"/></div><br>

The Taylor Series Expansion of *f(x)* at *x=∞* is given by: <br>
<div style="text-align:center"><img src ="Observations/series2.png"  width="500"/></div><br>

Minimum of *f(x)* is observed to be ≈-0.30884 at *x*≈-1.1924<br>

When visualized, Mish Activation Function closely resembles the function path of Swish having a small decay (preserve) in the negative side while being near linear on the positive side. It is a Monotonic Function and as observed from it's derivatives functions shown above and graph shown below, it can be noted that it has a Non-Monotonic 1<sup>st</sup> derivative and 2<sup>nd</sup> derivative. <br>

**Mish** ranges between ≈-0.31 to ∞.<br>
<div style="text-align:center"><img src ="Observations/Mish3.png"  width="800"/></div>
<div style="text-align:center"><img src ="Observations/Derivatives.png"  width="800"/></div>
Based on mathematical analysis, it is also confirmed that the function has a parametric order of continuity of: C<sup>∞</sup>

**Mish** has a very sharp global minima similar to Swish, which might account to gradients updates of the model being stuck in the region of sharp decay thus may lead to bad performance levels as compared to ReLU. Mish, also being mathematically heavy, is more computationally expensive as compared to the time complexity of Swish Activation Function. 

## Try It! 

### Demo Jupyter Notebooks:

All demo jupyter notebooks are presemt in the [Notebooks Folder](https://github.com/digantamisra98/Mish/tree/master/Notebooks)

### For Source Code Implementation: 

#### Torch:

#### Keras:

#### Tensorflow:

## Conclusion:

## Future Work (Coming Soon):

## Support Me

## Contact: 
- [LinkedIn](https://www.linkedin.com/in/misradiganta/)<br>
- Email: mishradiganta91@gmail.com
