<p align="center">
  <img width="300" src="Observations/Activation Function.png">
</p>

[![Donate](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![HitCount](http://hits.dwyl.io/digantamisra98/Mish.svg)](http://hits.dwyl.io/digantamisra98/Mish)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/digantamisra98/Mish/issues)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/DigantaMisra?locale.x=en_GB)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f67cf4cf73bf47fbbe4b3dd6f78bb10b)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=digantamisra98/Mish&amp;utm_campaign=Badge_Grade)
[![CircleCI](https://circleci.com/gh/digantamisra98/Mish.svg?style=svg)](https://circleci.com/gh/digantamisra98/Mish)
[![Paper](https://img.shields.io/badge/Paper-arXiv-blue.svg)](https://arxiv.org/abs/1908.08681)

# Mish: Self Regularized Non-Monotonic Activation Function

## Read the paper here - https://arxiv.org/abs/1908.08681

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
<div style="text-align:center"><img src ="Observations/imgtemp_ugysxo-1.png"  width="200"/></div>
It can also be represented by using the SoftPlus Activation Function as shown:<br><br>
<div style="text-align:center"><img src ="Observations/imgtemp_x5rglu-1.png"  width="150"/></div>
<div style="text-align:center"><img src ="Observations/imgtemp_utahjs-1.png"  width="270"/></div><br>
And it's 1<sup>st</sup> and 2<sup>nd</sup> derivatives are given below:<br>
<div style="text-align:center"><img src ="Observations/d1.png"  width="120"/></div>
<div style="text-align:center"><img src ="Observations/d2.png"  width="650"/></div><br>
Where:<br>
<div style="text-align:center"><img src ="Observations/delta.png"  width="150"/></div>
<div style="text-align:center"><img src ="Observations/omega.png"  width="350"/></div>
<br>

The Taylor Series Expansion of *f(x)* at *x=0* is given by: <br>
<div style="text-align:center"><img src ="Observations/series.png"  width="500"/></div><br>

The Taylor Series Expansion of *f(x)* at *x=∞* is given by: <br>
<div style="text-align:center"><img src ="Observations/series2.png"  width="350"/></div><br>

Minimum of *f(x)* is observed to be ≈-0.30884 at *x*≈-1.1924<br>

When visualized, Mish Activation Function closely resembles the function path of Swish having a small decay (preserve) in the negative side while being near linear on the positive side. It is a Non-Monotonic Function and as observed from it's derivatives functions shown above and graph shown below, it can be noted that it has a Non-Monotonic 1<sup>st</sup> derivative and 2<sup>nd</sup> derivative. <br>

**Mish** ranges between ≈-0.31 to ∞.<br>
<div style="text-align:center"><img src ="Observations/Mish3.png"  width="800"/></div>
<div style="text-align:center"><img src ="Observations/Derivatives.png"  width="800"/></div>

Following image shows the effect of Mish being applied on random noise. This is a replication of the effect of the activation function on the image tensor inputs in CNN models. 

<div style="text-align:center"><img src ="Observations/Mish_noise.png"  width="800"/></div>

Based on mathematical analysis, it is also confirmed that the function has a parametric order of continuity of: C<sup>∞</sup>

**Mish** has a very sharp global minima similar to Swish, which might account to gradients updates of the model being stuck in the region of sharp decay thus may lead to bad performance levels as compared to ReLU. Mish, also being mathematically heavy, is more computationally expensive as compared to the time complexity of Swish Activation Function. 

The output landscape of 5 layer randomly initialized neural network was compared for ReLU, Swish, and Mish. The observation clearly shows the sharp transition between the scalar magnitudes for the co-ordinates of ReLU as compared to Swish and Mish. Smoother transition results in smoother loss functions which are easier to optimize and hence the network generalizes better. Additional comparison of output landscapes is done for GELU, SELU, ELU, Leaky ReLU, PReLU and RReLU. Most of them similar to ReLU have sharp transitions in the output landscape and thus prove to be a roadblock to effective optimization of gradients. 

<div style="text-align:center"><img src ="Observations/Mish_Landscape_1.png"  width="800"/></div>
<div style="text-align:center"><img src ="Observations/comp123.png"  width="800"/></div>
<div style="text-align:center"><img src ="Observations/comp1234.png"  width="800"/></div>

The Pre-Activations (ωx + b) distribution was observed for the final convolution layer in a ResNet v1-20 with Mish activation function before and after training for 20 epochs on CIFAR-10. As shown below, units are being preserved in the negative side which improves the network capacity to generalize well due to less loss of information. 

<div style="text-align:center"><img src ="Observations/Distribution.png"  width="800"/></div>

Complex Analysis of Mish Activation Function: 

<div style="text-align:center"><img src ="Observations/complex.png"  width="800"/></div>

## Variation of Parameter Comparison:

### MNIST:

To observe how increasing the number of layers in a network while maintaining other parameters constant affect the test accuracy, fully connected networks of varying depths on MNIST, with each layer having 500 neurons were trained. Residual Connections were not used because they enable the training of arbitrarily deep networks. BatchNorm was used to lessen the dependence on initialization along with a dropout of 25%. The network is optimized using SGD on a batch size of 128, and for fair comparison, the same learning rates for each activation function was maintained. In the experiments, all 3 activations maintained nearly the same test accuracy for 15 layered Network. Increasing number of layers from 15 gradually resulted in a sharp decrease in test accuracy for Swish and ReLU, however, Mish outperformed them both in large networks where optimization becomes difficult.

The consistency of Mish providing better test top-1 accuracy as compared to Swish and ReLU was also observed by increasing Batch Size for a ResNet v2-20 on CIFAR-10 for 50 epochs while keeping all other network parameters to be constant for fair comparison.

<p float="left">
  <img src="Observations/layersacc.png"  width="420"/>
  <img src="Observations/batchacc.png"  width="420"/> 
</p>

Gaussian Noise with varying standard deviation was added to the input in case of MNIST classification using a simple conv net to observe the trend in decreasing test top-1 accuracy for Mish and compare it to that of ReLU and Swish. Mish mostly maintained a consistent lead over that of Swish and ReLU (Less than ReLU in just 1 instance and less than Swish in 3 instance) as shown below. The trend for test loss was also observed following the same procedure. (Mish has better loss than both Swish and ReLU except in 1 instance)

<p float="left">
  <img src="Observations/noise.png"  width="420"/>
  <img src="Observations/noise1.png"  width="420"/> 
</p>

The effect of various Optimizers on the Test Top-1 Accuracy of a simple 4 layered Conv Net with Mish on MNIST was visualized and compared against Swish. Mish had a better accuracy in 7 out of the 9 optimizers as shown below. Mish was also tested for different Learning Rates for *SGD* optimizer on MNIST and compared to Swish. The comparison confirms that Mish performs best on lower learning rates as compared to Swish. 

<p float="left">
  <img src="Observations/optim.png"  width="420"/>
  <img src="Observations/lr.png"  width="420"/>
</p>

The effect of various Weight initializers and Regularizers on the Test Top-1 Accuracy in the fully connected Dense Layer of a simple 4 layered Conv Net with Mish on MNIST was compared to that with Swish and the plots beneath shows that Mish has a significant improvement over Swish. 

<p float="left">
  <img src="Observations/init.png"  width="420"/>
  <img src="Observations/l1l2.png"  width="420"/>
</p>

The effect of increasing dropout rates and increasing dense units on Test Top-1 Accuracy for a 4 layered network using Mish on MNIST was compared to Swish. The graphs below show the consistency of Mish over Swish.

<p float="left">
  <img src="Observations/drop.png"  width="420"/>
  <img src="Observations/dense.png"  width="420"/>
</p>

### CIFAR10:

<p float="left">
  <img src="Observations/dropc10.png"  width="420"/>
  <img src="Observations/densec10.png"  width="420"/>
</p>

<p float="left">
  <img src="Observations/initc10.png"  width="420"/>
  <img src="Observations/regc10.png"  width="420"/>
</p>

<p float="left">
  <img src="Observations/lrc10.png"  width="420"/>
  <img src="Observations/augc10.png"  width="420"/>
</p>

<p float="left">
  <img src="Observations/optimc10.png"  width="420"/>
  <img src="Observations/policyan.png"  width="420"/>
</p>

*All default parameters were used for Optimizers.* <br>
*For Cosine Annealing, Max η was set at 0.01 (1e-2) and Min η was set at 0.0001 (1e-4)* <br>
*For One Cycle Policy, Min Learning Rate was set at 0.00000291545 (7e-3), Max Learning Rate was set at 0.00020408163 (7e-2), Min Momentum was set at 0.85, Max Momentum was set at 0.95, Annealing Stage was set at 0.1 and Annealing Rate was set at 0.01.*

<p float="left">
  <img src="Observations/mix1.png"  width="420"/>
  <img src="Observations/mix2.png"  width="420"/>
</p>

## Edge of Chaos and Rate of Convergence (EOC & ROC)/ Hessian Energy Computation Analysis: 

**Coming Soon**

## Significance Level: 

The P-values were computed for different activation functions in comparison to that of Mish on terms of Top-1 Testing Accuracy of a Squeeze Net Model on CIFAR-10 for 50 epochs for 3 runs and 23 runs using Adam Optimizer at a Learning Rate of 0.001 and Batch Size of 128. It was observed that Mish beats most of the activation functions at a high significance level in the 3 runs while for 23 runs, it beats ReLU at a high significance of P < 0.0001. Mish also had a comparatively lower standard deviation across both 3 and 23 runs which proves the consistency of performance for Mish.

### Sample Size = 3:

|Activation Function| Peak Accuracy | Mean Accuracy | Standard Deviation of Accuracy | P-value | Mean Loss|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Mish|**88.15%**|**87.93%**|0.04358898943540784|-|**4.018666666666666%**|
|ReLU|87.47%|87.06%|0.5311308689955831|P < 5e-2 (0.0475)|4.2956666666666665%|
|Swish-1|87.88%|87.36333333333333%|0.135030860670192|P < 5e-3 (0.0023)|4.191%|
|ELU(α=1.0)|86.82%|86.46333333333334%|0.07571877794400171|P < 0.0001|4.2090000000000005%|
|E-Swish (β=1.75)|87.92%|87.53999999999999%|0.33421549934136363|P < 5e-1 (0.1156)| 4.091333333333333%|
|GELU|87.89%|87.28%|0.15620499351812658|P < 5e-3 (0.0023)|4.405666666666667%|
|HardShrink(λ = 0.5)|75.44%|74.89333333333333%|0.6035174672976259|P < 0.0001|7.278333333333333%|
|Hardtanh|83.39%|82.79666666666667%|0.36963946398258196|P < 0.0001|5.132333333333333%|
|Leaky ReLU(α=0.3)|87.27%|87.06666666666666%|0.06429100507328683|P < 0.0001|4.067333333333333%|
|LogSigmoid|84.54%|82.41666666666667%|0.7203702751594688|P < 5e-4 (0.0002)|5.436%|
|PReLU|85.82%|84.61666666666666%|0.4534681172181107|P < 5e-4 (0.0002)|5.366666666666666%|
|RReLU|87.88%|86.82333333333334%|1.1430806329097392|P < 5e-1 (0.1691)|4.103666666666666%|
|ReLU6|87.5%|87.02333333333333%|0.092915732431772|P < 5e-4 (0.0001)|4.202333333333334%|
|SELU|84.54%|84.53666666666666%|0.26388128644020004|P < 0.0001|4.612666666666667%|
|CELU(α=1.0)|87.2%|86.52%|0.32969683043669107|P < 5e-3 (0.0018)| 4.145666666666667%|
|Sigmoid|81.75%|78.96%|1.8929606440705533|P < 5e-3 (0.0012)|6.463666666666667%|
|SoftPlus(β = 1)|84.93%|81.92333333333333%|1.6565727672919628|P < 5e-3 (0.0033)|6.008666666666667%|
|Tanhshrink|84.71%|83.63%|0.9457272334029457|P < 5e-3 (0.0014)|5.002666666666666%|
|Tanh|84.2%|83.41%|0.7397972695272689|P <= 5e-4 (0.0005)|5.053%|
|Softshrink(λ = 0.5)|83.34%|82.51666666666667%|0.22722969289539155|P < 0.0001|5.494666666666666%|
|Softsign|83.64%|83.23333333333333%|0.4398105652816147|P = 0.0001|5.056666666666667%|
|Aria-2(β = 1, α=1.5)|83.89%|82.67666666666666%|1.3052330570949109|P < 5e-3  (0.0022)|6.205666666666667%|
|Bent's Identity|85.66%|85.19666666666666%|0.3500476158086701|P < 5e-4 (0.0002)|4.434333333333333%|
|SQNL|83.72%|83.52%|0.20000000000000284|P < 0.0001|5.045%|
|ELisH|87.89%|87.86%|**0.04358898943540458**|P < 5e-1 (0.1206)|4.138%|
|Hard ELisH|86.85%|86.29%|0.11789826122551722|P < 5e-4 (0.0001)|4.2967%|
|SReLU|85.91%|85.347%|0.5600297611139322|P < 5e-3  (0.0013)|4.479%|
|ISRU (α=1.0)|84.14%|82.86%|0.7396170180122467|P < 5e-4 (0.0003)|5.335%|
|Flatten T-Swish|87.35%|86.85%|0.11060440015357959|P < 5e-4 (0.0001)|4.669%|

### Sample Size = 23:

|Activation Function| Mean Accuracy | Mean Loss| Standard Deviation of Accuracy | P-value | Cohen's d Score | 95% CI|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Mish|87.48%|4.13%|0.3967|-|-|-|
|Swish-1|87.32%|4.22%|0.414|P = 0.1973|0.386|-0.3975 to 0.0844|
|E-Swish (β=1.75)|**87.49%**|4.156%|0.411|P = 0.9075|0.034444|-0.2261 to 0.2539|
|GELU|87.37%|4.339%|0.472|P = 0.4003|0.250468|-0.3682 to 0.1499|
|ReLU|86.66%|4.398%|0.584|P < 0.0001|1.645536|-1.1179 to -0.5247|
|ELU(α=1.0)|86.41%|4.211%|**0.3371**|P < 0.0001|2.918232|-1.2931 to -0.8556|
|Leaky ReLU(α=0.3)|86.85%|**4.112%**|0.4569|P < 0.0001|1.47632|-0.8860 to -0.3774|
|RReLU|86.87%|4.138%|0.4478|P < 0.0001|1.444091|-0.8623 to -0.3595|
|SELU|83.91%|4.831%|0.5995|P < 0.0001|7.020812|-3.8713 to -3.2670|
|SoftPlus(β = 1)|83.004%|5.546%|1.4015|P < 0.0001|4.345453|-4.7778 to -4.1735|
|HardShrink(λ = 0.5)|75.03%|7.231%|0.98345|P < 0.0001|16.601747|-12.8948 to -12.0035|
|Hardtanh|82.78%|5.209%|0.4491|P < 0.0001|11.093842|-4.9522 to -4.4486|
|LogSigmoid|81.98%|5.705%|1.6751|P < 0.0001|4.517156|-6.2221 to -4.7753|
|PReLU|85.66%|5.101%|2.2406|P = 0.0004|1.128135|-2.7715 to -0.8590|
|ReLU6|86.75%|4.355%|0.4501|P < 0.0001| 1.711482|-0.9782 to -0.4740|
|CELU(α=1.0)|86.23%|4.243%|0.50941|P < 0.0001| 2.741669|-1.5231 to -0.9804|
|Sigmoid|74.82%|8.127%|5.7662|P < 0.0001|3.098289|-15.0915 to -10.2337|
|Softshrink(λ = 0.5)|82.35%|5.4915%|0.71959|P < 0.0001|8.830541|-5.4762 to -4.7856|
|Tanhshrink|82.35%|5.446%|0.94508|P < 0.0001|7.083564|-5.5646 to -4.7032|
|Tanh|83.15%|5.161%|0.6887|	P < 0.0001|7.700198|-4.6618 to -3.9938|
|Softsign|82.66%|5.258%|0.6697|P < 0.0001|8.761157|-5.1493 to -4.4951|
|Aria-2(β = 1, α=1.5)|81.31%|6.0021%|2.35475|P < 0.0001|3.655362|-7.1757 to -5.1687|
|Bent's Identity|85.03%|4.531%|0.60404|P < 0.0001|4.80211|-2.7576 to -2.1502|
|SQNL|83.44%|5.015%|0.46819|P < 0.0001|9.317237|-4.3009 to -3.7852|
|ELisH|87.38%|4.288%|0.47731|P = 0.4283|0.235784|-0.3643 to 0.1573|
|Hard ELisH|85.89%|4.431%|0.62245|P < 0.0001|3.048849|-1.9015 to -1.2811|
|SReLU|85.05%|4.541%|0.5826|P < 0.0001|4.883831|-2.7306 to -2.1381|

*Values rounded up which might cause slight deviation in the statistical values reproduced from these tests*

### Confidence Interval Profiles: 

|Activation Function| CI |
|:---:|:---:|
|Mish|87.48 ± 0.1716|
|Swish-1|87.32347 ± 0.179027|
|E-Swish (β=1.75)|87.49391 ± 0.1776597|
|GELU|87.37083 ± 0.2040073|
|ReLU|86.65869 ± 0.2524601|
|ELU(α=1.0)|86.40565 ± 0.1458006|
|Leaky ReLU(α=0.3)|86.84826 ± 0.1976138|
|RReLU| 86.86913 ± 0.1936264|
|SELU|83.91086 ± 0.2592722|
|SoftPlus(β = 1)|83.00434 ± 0.6060631|
|HardShrink(λ = 0.5)|75.03086 ± 0.4252852|
|Hardtanh|82.77956 ± 0.1941855|
|LogSigmoid|81.9813 ± 0.7244|
|PReLU|85.66478 ± 0.968944|
|ReLU6|86.75391 ± 0.1946326|
|CELU(α=1.0)|86.22826 ± 0.2202884|
|Sigmoid|74.81739 ± 2.4934984|
|Softshrink(λ = 0.5)|82.34913 ± 0.3111762|
|Tanhshrink|82.34608 ± 0.4086837|
|Tanh|83.15217 ± 0.2978422|
|Softsign|82.65782 ± 0.2896004|
|Aria-2(β = 1, α=1.5)|81.30782 ± 1.0182716|
|Bent's Identity|85.02608 ± 0.2612082|
|SQNL|83.43695 ± 0.2024614|
|ELisH|87.37652 ± 0.2064078|
|Hard ELisH|85.88869 ± 0.2691689|
|SReLU|85.04565 ± 0.2519697|

<div style="text-align:center"><img src ="Observations/ci1.png"  width="1000"/></div>

## Properties Summary:

|Activation Function Name| Function Graph | Equation | Range | Order of Continuity | Monotonic | Monotonic Derivative | Approximates Identity Near Origin| Dead Neurons | Saturated |
|---|---|---|---|---|---|---|---|---|---|
|Mish|<div style="text-align:center"><img src ="Observations/graph_mish.png"  width="500"/></div>|<div style="text-align:center"><img src ="Observations/table_eq.png"  width="700"/></div>| ≈-0.31 to ∞| C<sup>∞</sup> | No :negative_squared_cross_mark:| No :negative_squared_cross_mark: | Yes :heavy_check_mark: (Approximates half of identity at origin) | No :negative_squared_cross_mark: | No :negative_squared_cross_mark: |

## Results:

All results and comparative analysis are present in the [Readme](https://github.com/digantamisra98/Mish/blob/master/Notebooks/Readme.md) file present in the [Notebooks Folder](https://github.com/digantamisra98/Mish/tree/master/Notebooks).

### Summary of Results: 

*Comparison is done based on the high priority metric, for image classification the Top-1 Accuracy while for Generative Networks and Image Segmentation the Loss Metric. Therefore, for the latter, Mish > Baseline is indicative of better loss and vice versa. For Embeddings, the AUC metric is considered.*

|Activation Function| Mish > Baseline Model | Mish < Baseline Model |
|---|---|---|
|ReLU|55|20|
|Swish-1|53|22|
|SELU|21|1|
|TanH|19|0|
|Sigmoid|19|0|
|Softsign|19|1|
|ELU(α=1.0)|19|4|
|Tanhshrink|18|0|
|HardShrink(λ = 0.5)|18|0|
|Softshrink (λ = 0.5)|18|0|
|PReLU(Default Parameters)	|18|2|
|E-Swish (β=1.75)|18|6|
|Hardtanh|17|1|
|GELU|17|2|
|CELU(α=1.0)|16|2|
|LogSigmoid|16|3|
|SoftPlus(β = 1)|15|4|
|ReLU6|14|5|
|Leaky ReLU(α=0.3)|13|7|
|RReLU|11|8|
|Aria-2(β = 1, α=1.5)|10|1|
|SQNL|8|0|
|Bent's Identity|6|2|
|ELisH|4|0|
|Weighted TanH (Weight = 1.7145)|4|0|
|Hard ELisH|4|0|
|SineReLU (ε = 0.001)|4|0|
|Flatten T-Swish|4|0|
|ISRU (α=1.0)|4|0|
|Le Cun's TanH|4|0|
|Soft Clipping (α=0.5)|4|0|
|ISRLU (α=1.0)|2|2|
|SReLU|2|2|
|Hard Sigmoid|1|0|
|Thresholded ReLU(θ=1.0)|1|0|

#### Sample Result: 

|Configurations|Parameters|
|:---:|:---:|
|Model|Squeeze Excite ResNet-50 (SENet-50)|
|Dataset|CIFAR-10|
|Batch Size| 128|
|Epoch|100|
|Optimizer|Adam|
|Learning Rate|0.001|

|Activation Function | Testing Top-1 Accuracy|Loss|Testing Top-3 Accuracy|
|---|---|---|---|
|Mish|**90.7931%**|**4.75271%**|98.5562%|
|Swish-1|90.558%|4.76047%|98.6748%|
|E-Swish (β = 1.75)|90.5063%|5.22954%|98.6946%|
|ReLU|90.447%|4.93086%|98.6155%|
|GELU|90.5063%|5.0612%|**98.754%**|
|SELU|86.432%|6.89385%|97.8936%|

<div style="text-align:center"><img src ="Observations/se50_1.png"  width="1000"/></div>

It was observed that the stability of descent of Loss for SENet-50 with Mish is much better as compared to other activation functions. It was also observed that Mish was the only activation function which crossed the 91% mark for the Test Top-1 accuracy across both the runs while others reached a maximum of 90.7% with Mish recording the highest at 91.248%. 

Note - The graph represents the Test Top-1 accuracy and loss. Training Top-1 Accuracy and Loss are represented using dashed lines. 

#### CIFAR Results (Top-1 Testing Accuracy):

|Models|Mish|Swish|ReLU|
|:---:|:---:|:---:|:---:|
|ResNet v1-20| <p>CIFAR 10: 91.81%<br>CIFAR 100: <b>67.26%</b></p> | <p>CIFAR 10: <b>91.95%</b><br>CIFAR 100: 67.1%</p> | <p>CIFAR 10: 91.5%<br>CIFAR 100: 67%</p> |
|ResNet v1-32| <p>CIFAR 10: 92.29%<br>CIFAR 100: <b>69.44%</b></p> | <p>CIFAR 10: <b>92.3%</b><br>CIFAR 100: 68.84%</p> | <p>CIFAR 10: 91.78%<br>CIFAR 100: 68.45%</p> |
|ResNet v1-44|<p>CIFAR 10: 92.46%<br>CIFAR 100: 69.34%</p> | <p>CIFAR 10: <b>92.84%</b><br>CIFAR 100: 69.62%</p> | <p>CIFAR 10: 92.33%<br>CIFAR 100: <b>69.73%</b></p> |
|ResNet v1-56|<p>CIFAR 10: <b>92.21%</b><br>CIFAR 100: <b>70.13%</b></p> | <p>CIFAR 10: 91.85%<br>CIFAR 100: 70.02%</p> | <p>CIFAR 10: 91.97%<br>CIFAR 100: 69.6%</p> |
|ResNet v1-110|<p>CIFAR 10: 91.44%<br>CIFAR 100: 67.64%</p> | <p>CIFAR 10: 91.34%<br>CIFAR 100: 67.76%</p> | <p>CIFAR 10: <b>91.69%</b><br>CIFAR 100: </b>68.43%</b></p> |
|ResNet v1-164|<p>CIFAR 10: <b>83.62%</b><br>CIFAR 100: 52.7%</p> | <p>CIFAR 10: 82.19%<br>CIFAR 100: <b>55.96%</b></p> | <p>CIFAR 10: 82.37%<br>CIFAR 100: 52.6%</p> |
|ResNet v2-20|<p>CIFAR 10: <b>92.02%</b><br>CIFAR 100: <b>70.86%</b></p> | <p>CIFAR 10: 91.61%<br>CIFAR 100: 70.23%</p> | <p>CIFAR 10: 91.71%<br>CIFAR 100: 70.54%</p> |
|ResNet v2-56|<p>CIFAR 10: <b>87.18%</b><br>CIFAR 100: <b>60.67%</b></p> | <p>CIFAR 10: 86.36%<br>CIFAR 100: 60.28%</p> | <p>CIFAR 10: 83.86%<br>CIFAR 100: 57.25%</p> |
|ResNet v2-110|<p>CIFAR 10: <b>92.58%</b><br>CIFAR 100: <b>74.41%</b></p> | <p>CIFAR 10: 92.22%<br>CIFAR 100: 74.13%</p> | <p>CIFAR 10: 91.93%<br>CIFAR 100: 73%</p> |
|ResNet v2-164|<p>CIFAR 10: <b>87.74%</b><br>CIFAR 100: 64.16%</p> | <p>CIFAR 10: 86.13%<br>CIFAR 100: <b>64.48%</b></p> | <p>CIFAR 10: 83.59%<br>CIFAR 100: 63.73%</p> |
|ResNet v2-245|<p>CIFAR 10: <b>86.67%</b><br>CIFAR 100: NA</p> | <p>CIFAR 10: 85.41%<br>CIFAR 100: NA</p> | <p>CIFAR 10: 86.32%<br>CIFAR 100: NA</p> |
|WRN 10-2|<p>CIFAR 10: <b>86.83%</b><br>CIFAR 100: <b>67.157%</b></p> | <p>CIFAR 10: 86.56%<br>CIFAR 100: 66.98%</p> | <p>CIFAR 10: 84.52%<br>CIFAR 100: 62.5567%</p> |
|WRN 16-4|<p>CIFAR 10: 90.54%<br>CIFAR 100: 72.92%</p> | <p>CIFAR 10: 90.07%<br>CIFAR 100: <b>74.60%</b></p> | <p>CIFAR 10: <b>90.74%</b><br>CIFAR 100: <b>74.60%</b></p> |
|WRN 22-10|<p>CIFAR 10: 90.38%<br>CIFAR 100: <b>72.32%</b></p> | <p>CIFAR 10: 90.17%<br>CIFAR 100: 71.89%</p> | <p>CIFAR 10: <b>91.28%</b><br>CIFAR 100: 72.2%</p> |
|WRN 40-4|<p>CIFAR 10: NA<br>CIFAR 100: 69.52%</p> | <p>CIFAR 10: NA<br>CIFAR 100: <b>69.59%</b></p> | <p>CIFAR 10: NA<br>CIFAR 100: 69.35%</p> |
|VGG 16|<p>CIFAR 10: NA<br>CIFAR 100: 68.64%</p> | <p>CIFAR 10: NA<br>CIFAR 100: <b>69.7%</b></p> | <p>CIFAR 10: NA<br>CIFAR 100: 69.36%</p> |
|SimpleNet|<p>CIFAR 10: <b>91.70%</b><br>CIFAR 100: NA</p> | <p>CIFAR 10: 91.44%<br>CIFAR 100: NA</p> | <p>CIFAR 10: 91.16%<br>CIFAR 100: NA</p> |
|Xception Net|<p>CIFAR 10: <b>88.73%</b><br>CIFAR 100: NA</p> | <p>CIFAR 10: 88.56%<br>CIFAR 100: NA</p> | <p>CIFAR 10: 88.38%<br>CIFAR 100: NA</p> |
|Capsule Network|<p>CIFAR 10: <b>83.15%</b><br>CIFAR 100: NA</p> | <p>CIFAR 10: 82.48%<br>CIFAR 100: NA</p> | <p>CIFAR 10: 82.19%<br>CIFAR 100: NA</p> |
|Inception ResNet v2|<p>CIFAR 10: <b>85.21%</b><br>CIFAR 100: NA</p> | <p>CIFAR 10: 84.96%<br>CIFAR 100: NA</p> | <p>CIFAR 10: 82.22%<br>CIFAR 100: NA</p> |
|DenseNet-121|<p>CIFAR 10: <b>91.2678%</b><br>CIFAR 100: <b>66.3172%</b></p> | <p>CIFAR 10: 90.9217%<br>CIFAR 100: 65.9118%</p> | <p>CIFAR 10: 91.0997%<br>CIFAR 100: 65.5063%</p> |
|DenseNet - 161|<p>CIFAR 10: 90.8228%<br>CIFAR 100: 63.8944%</p> | <p>CIFAR 10: 90.1602%<br>CIFAR 100: <b>64.8042%</b></p> | <p>CIFAR 10: <b>91.0206%</b><br>CIFAR 100: 63.9043%</p> |
|DenseNet - 169|<p>CIFAR 10: 90.5063%<br>CIFAR 100: 65.3877%</p> | <p>CIFAR 10: 90.6744%<br>CIFAR 100: <b>65.6942%</b></p> | <p>CIFAR 10: <b>91.6535%</b><br>CIFAR 100: 64.9921%</p> |
|DenseNet - 201|<p>CIFAR 10: 90.7338%<br>CIFAR 100: <b>64.4383%</b></p> | <p>CIFAR 10: <b>91.0107%</b><br>CIFAR 100: 64.2504%</p> | <p>CIFAR 10: 90.7239%<br>CIFAR 100: 63.2516%</p> |
|ResNext - 50|<p>CIFAR 10: 90.8327%<br>CIFAR 100: <b>67.5831%</b></p> | <p>CIFAR 10: <b>91.6238%</b><br>CIFAR 100: 66.7227%</p> | <p>CIFAR 10: 89.3592%<br>CIFAR 100: 67.5237%</p> |
|MobileNet v1|<p>CIFAR 10: 85.2749%<br>CIFAR 100: <b>51.93%</b></p> | <p>CIFAR 10: <b>85.6903%</b><br>CIFAR 100: 49.9506%</p> | <p>CIFAR 10: 84.1179%<br>CIFAR 100: 49.2089%</p> |
|MobileNet v2|<p>CIFAR 10: <b>86.254%</b><br>CIFAR 100: <b>57.0609%</b></p> | <p>CIFAR 10: 86.0759%<br>CIFAR 100: 55.6764%</p> | <p>CIFAR 10: 86.0463%<br>CIFAR 100: 56.1907%</p> |
|SENet - 18|<p>CIFAR 10: <b>90.526%</b><br>CIFAR 100: <b>64.3888%</b></p> | <p>CIFAR 10: 89.4284%<br>CIFAR 100: 63.8944%</p> | <p>CIFAR 10: 90.1602%<br>CIFAR 100: 62.7176%</p> |
|SENet - 34|<p>CIFAR 10: 91.0997%<br>CIFAR 100: 64.4778%</p> | <p>CIFAR 10: 89.9624%<br>CIFAR 100: <b>64.8734%</b></p> | <p>CIFAR 10: <b>91.6733%</b><br>CIFAR 100: 64.5669%</p> |
|SENet - 50|<p>CIFAR 10: <b>90.7931%</b><br>CIFAR 100: NA</p> | <p>CIFAR 10: 90.5558%<br>CIFAR 100: NA</p> | <p>CIFAR 10: 90.447%<br>CIFAR 100: NA</p> |
|ShuffleNet v1|<p>CIFAR 10: <b>87.3121%</b><br>CIFAR 100: <b>59.1871%</b></p> | <p>CIFAR 10: 86.9462%<br>CIFAR 100: 58.4355%</p> | <p>CIFAR 10: 87.0451%<br>CIFAR 100: 57.9806%</p> |
|ShuffleNet v2|<p>CIFAR 10: <b>87.37%</b><br>CIFAR 100: <b>59.3552%</b></p> | <p>CIFAR 10: 86.9363%<br>CIFAR 100: 58.9102%</p> | <p>CIFAR 10: 87.0055%<br>CIFAR 100: 58.5641%</p> |
|SqueezeNet|<p>CIFAR 10: 88.13%<br>CIFAR 100: <b>63.0736%</b></p> | <p>CIFAR 10: <b>88.3703%</b><br>CIFAR 100: 62.1143%</p> | <p>CIFAR 10: 87.8461%<br>CIFAR 100: 60.9276%</p> |
|Inception v3|<p>CIFAR 10: <b>91.8314%</b><br>CIFAR 100: 68.3347%</p> | <p>CIFAR 10: 91.1788%<br>CIFAR 100: 67.0095%</p> | <p>CIFAR 10: 90.8426%<br>CIFAR 100: <b>68.7797%</b></p> |
|Efficient Net B0|<p>CIFAR 10: <b>80.7358%</b><br>CIFAR 100: NA</p> | <p>CIFAR 10: 79.371%<br>CIFAR 100: NA</p> | <p>CIFAR 10: 79.3117%<br>CIFAR 100: NA</p> |
|Efficient Net B1|<p>CIFAR 10: 80.9632%<br>CIFAR 100: NA</p> | <p>CIFAR 10: 81.9818%<br>CIFAR 100: NA</p> | <p>CIFAR 10: <b>82.4367%</b><br>CIFAR 100: NA</p> |
|Efficient Net B2|<p>CIFAR 10: 81.2006%<br>CIFAR 100: NA</p> | <p>CIFAR 10: 80.9039%<br>CIFAR 100: NA</p> | <p>CIFAR 10: <b>81.7148%</b><br>CIFAR 100: NA</p> |


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

#### MXNet:

MXNet Implementation of Mish Activation function can be found [here](https://github.com/digantamisra98/Mish/blob/master/Mish/MXNet/mish.py)

## Future Work (Coming Soon):

- Additional STL-10, CalTech-101 & 256 Benchmarks.
- Image Net Benchmarks.
- GANs Benchmarks.
- Transformer Model Benchmarks.
- Fix ResNext Benchmarks.
- Comparison of Convergence Rates.

## Cite this work:

```
@misc{1908.08681,
Author = {Diganta Misra},
Title = {Mish: A Self Regularized Non-Monotonic Neural Activation Function},
Year = {2019},
Eprint = {arXiv:1908.08681},
}
```

## Contact: 
- [LinkedIn](https://www.linkedin.com/in/misradiganta/)<br>
- Email: mishradiganta91@gmail.com
