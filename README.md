<p align="center">
  <img width="200" src="Observations/logo.PNG">
</p>

[![HitCount](http://hits.dwyl.io/digantamisra98/Mish.svg)](http://hits.dwyl.io/digantamisra98/Mish)
[![Dependencies](https://david-dm.org/digantamisra98/Mish.svg)](http://david-dm.org/digantamisra98/Mish)

# Mish: Self Regularized Non-Monotonic Activation Function

Inspired by *Swish* Activation Function ([Paper](https://arxiv.org/abs/1710.05941)), **Mish** is a Self Regularized Non-Monotonic Neural Activation Function. Activation Function serves a core functionality in the training process of a Neural Network Architecture and is represented by the basic mathematical representation: 
<div style="text-align:center"><img src ="Observations/act.png"  width="370"/></div>
<br>
An Activation Function is generally used to introduce non-linearity and over the years of theoretical machine learning research, many activation functions have been constructed with the 2 most popular amongst them being: 

-ReLU (Rectified Linear Unit; f(x)=max(0,x)) <br>
-TanH <br>

Other notable ones being: <br> 
-Softmax (Used for Multi-class Classification in the output layer) <br> 
-Sigmoid (f(x)=(1+e<sup>-x</sup>)<sup>-1</sup>;Used for Binary Classification and Logistic Regression) <br>
-Leaky ReLU (f(x)=0.001x (x<0) or x (x>0)) <br>



## Dependencies
- TensorFlow = 1.12.x or higher
- Keras = 2.2.x or higher
- Python = 3x

## Mathematics under the hood:

Mish Activation Function can be mathematically represented by the following formula:<br> 
<div style="text-align:center"><img src ="Observations/imgtemp_ugysxo-1.png"  width="220"/></div><br>
And it's 1<sup>st</sup> and 2<sup>nd</sup> derivatives are given below:<br>
<div style="text-align:center"><img src ="Observations/imgtemp_8ipqjq-1.png"  width="200"/></div>
<div style="text-align:center"><img src ="Observations/imgtemp_qph7sj-1.png"  width="320"/></div><br>
Where:<br>
<div style="text-align:center"><img src ="Observations/imgtemp_lz642a-1.png"  width="200"/></div>
<br>
<div style="text-align:center"><img src ="Observations/imgtemp_3rbfba-1.png"  width="270"/></div>
<br>
<div style="text-align:center"><img src ="Observations/2b.png"  width="800"/></div>
<br>
<div style="text-align:center"><img src ="Observations/imgtemp_kyk9k1-1.png"  width="30"/></div>
<br>
