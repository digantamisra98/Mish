<p align="center">
  <img width="300" src="Observations/Activation Function.png">
</p>

[![Donate](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![HitCount](http://hits.dwyl.io/digantamisra98/Mish.svg)](http://hits.dwyl.io/digantamisra98/Mish)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/digantamisra98/Mish/issues)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/DigantaMisra?locale.x=en_GB)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f67cf4cf73bf47fbbe4b3dd6f78bb10b)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=digantamisra98/Mish&amp;utm_campaign=Badge_Grade)
[![CircleCI](https://circleci.com/gh/digantamisra98/Mish.svg?style=svg)](https://circleci.com/gh/digantamisra98/Mish)

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

When visualized, Mish Activation Function closely resembles the function path of Swish having a small decay (preserve) in the negative side while being near linear on the positive side. It is a Non-Monotonic Function and as observed from it's derivatives functions shown above and graph shown below, it can be noted that it has a Non-Monotonic 1<sup>st</sup> derivative and 2<sup>nd</sup> derivative. <br>

**Mish** ranges between ≈-0.31 to ∞.<br>
<div style="text-align:center"><img src ="Observations/Mish3.png"  width="800"/></div>
<div style="text-align:center"><img src ="Observations/Derivatives.png"  width="800"/></div>

Following image shows the effect of Mish being applied on random noise. This is a replication of the effect of the activtion function on the image tensor inputs in CNN models. 

<div style="text-align:center"><img src ="Observations/Mish_noise.png"  width="800"/></div>

Here, Mish activation function was applied on a standard normal distribution. The output distribution shows Mish to be preserving information of the original distribution in the negative axis. 

<div style="text-align:center"><img src ="Observations/Mish_bar.png"  width="800"/></div>

Based on mathematical analysis, it is also confirmed that the function has a parametric order of continuity of: C<sup>∞</sup>

**Mish** has a very sharp global minima similar to Swish, which might account to gradients updates of the model being stuck in the region of sharp decay thus may lead to bad performance levels as compared to ReLU. Mish, also being mathematically heavy, is more computationally expensive as compared to the time complexity of Swish Activation Function. 

The output landscape of 5 layer randomly initialized neural network was compared for ReLU, Swish and Mish. The observation clearly shows the sharp transition between the scalar magnitudes for the co-ordinates of ReLU as compared to Swish and Mish. Smoother transition results in smoother loss functions which are easier to optimize and hence the network generalizes better. 

<div style="text-align:center"><img src ="Observations/Mish_Landscape_1.png"  width="800"/></div>

The Pre-Activations (ωx + b) distribution was observed for the final convolution layer in a ResNet v1-20 with Mish activation function before and after training for 20 epochs on CIFAR-10. As shown below, units are being preserved in the negative side which improves the network capacity to generalize well due to less loss of information. 

<div style="text-align:center"><img src ="Observations/Distribution.png"  width="800"/></div>

A 9 layer Network was trained for 50 epochs on CIFAR-10 to visualize the Loss Contour and Weights Distribution Histograms by following Filter Normalization process: 

<div style="text-align:center"><img src ="Observations/Histogram_Mish.png"  width="800"/></div>
<div style="text-align:center"><img src ="Observations/Mish_loss_2d.png"  width="800"/></div>

Complex Analysis of Mish Activation Function: 

<div style="text-align:center"><img src ="Observations/complex.png"  width="800"/></div>

## Edge of Chaos and Rate of Convergence (EOC & ROC): 

**Coming Soon**

## Properties Summary:

|Activation Function Name| Function Graph | Equation | Range | Order of Continuity | Monotonic | Monotonic Derivative | Approximates Identity Near Origin| Dead Neurons | Saturated |
|---|---|---|---|---|---|---|---|---|---|
|Mish|<div style="text-align:center"><img src ="Observations/graph_mish.png"  width="500"/></div>|<div style="text-align:center"><img src ="Observations/table_eq.png"  width="700"/></div>| ≈-0.31 to ∞| C<sup>∞</sup> | No :negative_squared_cross_mark:| No :negative_squared_cross_mark: | Yes :heavy_check_mark:| No :negative_squared_cross_mark: | No :negative_squared_cross_mark: |

## Results:

All results and comparative analysis are present in the [Readme](https://github.com/digantamisra98/Mish/blob/master/Notebooks/Readme.md) file present in the [Notebooks Folder](https://github.com/digantamisra98/Mish/tree/master/Notebooks).

### Summary of Results: 

*Comparison is done based on the high priority metric, for image classification the Top-1 Accuracy while for Generative Networks and Image Segmentation the Loss Metric. Therefore, for the latter, Mish > Baseline is indicative of better loss and vice versa.*

|Activation Function| Mish > Baseline Model | Mish < Baseline Model |
|---|---|---|
|ReLU|41|19|
|Swish-1|40|20|
|ELU(α=1.0)|4|1|
|Aria-2(β = 1, α=1.5)|1|0|
|Bent's Identity|1|0|
|Hard Sigmoid|1|0|
|Leaky ReLU(α=0.3)|2|1|
|PReLU(Default Parameters)	|2|0|
|SELU|4|0|
|sigmoid|2|0|
|SoftPlus|1|0|
|Softsign|2|0|
|TanH|2|0|
|Thresholded ReLU(θ=1.0)|1|0|

## Try It! 

### Demo Jupyter Notebooks:

All demo jupyter notebooks are present in the [Notebooks Folder](https://github.com/digantamisra98/Mish/tree/master/Notebooks).

### For Source Code Implementation: 

#### Torch:

Torch Implementation of Mish Activation Function can be found [here](https://github.com/digantamisra98/Mish/tree/master/Mish/Torch)

#### Keras:

Keras Implementation of Mish activation function can be found [here](https://github.com/digantamisra98/Mish/blob/master/Mish/Keras/mish.py)

#### Tensorflow:

TensorFlow - Keras Implementation of Mish Activation function can be found [here](https://github.com/digantamisra98/Mish/blob/master/Mish/TF-Keras/mish.py)

## Future Work (Coming Soon):

- Additional CIFAR-100, STL-10, CalTech-101 Benchmarks
- Image Net Benchmarks
- GANs Benchmarks
- Transformer Model Benchmarks
- Fix ResNext Benchmarks
- Comparison of Convergence Rates

## Cite this work:

```
@misc{Mish: A Self Regularized Non-Monotonic Neural Activation Function},
  author = {Diganta Misra},
  title = {Mish: A Self Regularized Non-Monotonic Neural Activation Function},
  year = {2019},
  url = {https://github.com/digantamisra98/Mish},
}
```

## Contact: 
- [LinkedIn](https://www.linkedin.com/in/misradiganta/)<br>
- Email: mishradiganta91@gmail.com
