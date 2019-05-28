## Results:

During the research of Mish Activation Function, comparative study of Mish against Swish and ReLU was performed on datasets including MNIST, Fashion-MNIST, CIFAR10, CIFAR100, Caravan Challenge Dataset, ASL (American Sign Language), IRIS and some custom datasets including Malaria Cells Image Dataset using architectures including ResNet (v2-50), WRN (Wide Residual Networks, 10-2, 16-8, 28-10, 40-4), Mini VGG Net, LeNeT, Custom Deep CNN, ANN, SimpleNet, U-Net, DenseNet, etc.

### MNIST:
Google LeNet ([Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)) was used for MNIST - A database of Hand-written digits ([Dataset](http://yann.lecun.com/exdb/mnist/)) classification. The Accuracies table along with the inference time and computational cost analysis is provided below. The reason why LeNet was deployed for this task is because of the network being extremely small (Here, while referring to LeNet, it means LeNet-4 having a pair of Conv+Pool layers) and is extremely robust in MNIST classification, it also takes very less time to train due to the size of the network. 

| Activation Function  | Accuracy (20*) |  Loss (20*) | GPU-Utilization (20*) |CPU-RAM Utilization** (20*)| Training Time (20*) | Inference Time (20*)| Top 5 Accuracy (20*) | Top 3 Accuracy (20*)|
| ------------- | ------------- | ---|---|---|---|---|---|---|
| ReLU  | **98.65%**  |**0.368%**|5%|**11.4GB**|**51.67 seconds**|**0.197 seconds**|**100%**|**99.94%**|
| Swish  | 98.42%  |0.385%|5%|**11.4GB**|65.11 seconds|0.2157 seconds|99.99%|99.9%|
| Mish  | 98.64%  |**0.368%**|5%|11.2GB|81.12 seconds|0.2967 seconds|**100%**|**99.94%**|

<em> *The number indicates the Number of Epochs
</em><br>
<em> **This shows the amount of RAM Free.
</em><br>
The activation maps of the hidden layers were also visualized to understand the generalization the network was adopting to.

<div style="text-align:center"><img src ="Observations/blackbox.PNG"  width="700"/></div>
<br>

### Fashion-MNIST:

Mini VGG-Net ([Paper](https://arxiv.org/pdf/1409.1556.pdf)) was used for classification problem of Fashion MNIST/ F-MNIST ([Dataset](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)) which contains 28x28 sized images of fashion apparel.  The Accuracies table along with the inference time and computational cost analysis is provided below.

| Activation Function  | Accuracy (25*) |  Loss (25*) | GPU-Utilization (5*) |CPU-RAM Utilization** (5*)| Training Time (5*) | Inference Time (5*)|
| ------------- | ------------- | ---|---|---|---|---|
| ReLU  | 93.19%  |1.895%|33%|**10.1GB**|**261.88 seconds**|**3.51 seconds**|
| Swish  | 93.09%  |1.935%|33%|**10.1GB**|271.13 seconds|3.53 seconds|
| Mish  | **93.31%**|**1.859%**|33%|10GB|294.85 seconds|3.78 seconds|

<em> *The number indicates the Number of Epochs
</em><br>
<em> **This shows the amount of RAM Free.
</em><br>

The evaluation metrics for the Mini-VGG Network with Mish Activation Function is given below: 

| Class Labels  | Precision |  Recall | F1-Score |
| ------------- | ------------- | ---|---|
| top  | 0.90  |0.87|0.88|
| trouser  | 0.99 |0.98|0.99|
| pullover  | 0.93|0.89|0.91|
| dress  | 0.93 |0.94|0.93|
| coat | 0.88  |0.93|0.90|
| sandal | 0.99|0.99|0.99|
| shirt  | 0.79 |0.80|0.80|
| sneaker  | 0.96 |0.98|0.97|
| bag  | 0.99  |0.99|0.99|
| ankle-boot  | 0.98  |0.97|0.97|
| **Average** | **Precision** |**Recall** |**F1-Score**|
| micro average  | 0.93  |0.93|0.93|
| macro average  | 0.93  |0.93|0.93|
| weighted average  | 0.93  |0.93|0.93|


Test Samples obtained from the network:

<div style="text-align:center"><img src ="Observations/test.PNG"  width="300"/></div>
<br>

### Iris

A 3-layered Feed Forward Neural Network was used for IRIS ([Dataset](https://archive.ics.uci.edu/ml/datasets/iris)) classification. The metrics scores are provided in the table below. Here, in place of ReLU, Mish was tested against Swish and Sigmoid. 

| Activation Function  | Accuracy (4000*) |  Loss (4000*) |
| ------------- | ------------- | ---|
| ReLU  | 96.67%  |2.84%|
| Swish  | 97.33%  |**2.32%**|
| Mish  | **98%**|2.66%|

<em> *The number indicates the Number of Epochs
</em><br>

### CIFAR-10:

CIFAR-10 ([Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)) is an extensive labelled dataset of 60000, 32x32 Images belonging to 10 classes which have 6000 images per class. A comprehensive labelled subset of [TinyImages Dataset](http://groups.csail.mit.edu/vision/TinyImages/), the CIFAR-10 has been expansively used for benchmarking different architectures, frameworks and novel approaches mainly in the field of Image Classification. In this research, 3 primary kinds of Neural Networks have been used with varying parameters to test how Mish fares against Swish and ReLU. All of these details has been provided subsequently.

#### ResNet v2:

ResNet ([Paper](https://arxiv.org/abs/1512.03385)) v2 with 56 layers was used for CIFAR-10 classification task. Number of Epochs were varied to observe the computational cost and training time of the networks. The table below provides all the information regarding the same. 

- For Batch Size = 32, Number of Steps= 1563, Number of Epochs= 10:

|Activation Function |Training Accuracy|Training Loss|Validation Accuracy|Validation Loss|Testing Accuracy|Testing Loss|Average Per Epoch Time|Inference Time (Per Sample)|Average Per Step Time|Average Forward Pass Time|
|---|---|---|---|---|---|---|---|---|---|---|
|ReLU|73.10%|15.1%|71.9%|15.35%|73.34%|15.34%|**130.8 seconds**|**2 seconds (487 micro seconds)**|**83.8 milli seconds**|**669.9 micro- seconds**|
|Swish|**77.65%**|**14.04%**|75.58%|14.77%|75.88%|14.68%|155.1 seconds|3 seconds (550 micro-seconds)|99.3 milli-seconds|775.2 micro-seconds|
|Mish|76.93%|14.08%|**76.58%**|**14%**|**76.46%**|**13.98%**|158.5 seconds|3 seconds (590 micro-seconds)|101.4 milli-seconds|830.4 micro-seconds|

- For Batch Size = 32, Number of Steps= 1563, Number of Epochs= 50: (Best of 2 runs)

|Activation Function |Testing Accuracy|Testing Loss|Inference Time (Per Sample)| 
|---|---|---|---|
|ReLU|83.86%|9.945%|3 seconds (559 micro-seconds)|
|Swish|86.36%|8.81%|3 seconds (618 micro-seconds)|
|Mish|**87.18%**|**8.62%**|3 seconds (653 micro-seconds)|

<div style="text-align:center"><img src ="Observations/All_50.png"  width="1000"/></div>
<br>

- For Batch Size = 32, Number of Steps= 1563, Number of Epochs= 100: (Only *Mish*)

Additionally, ResNet v2 with Mish was also used for training on 100 epochs to confirm that the network doesn't face Gradient Death problem when epochs increases. The observations are provided in the table below:

|Training Accuracy|Training Loss|Validation Accuracy|Validation Loss|Testing Accuracy|Testing Loss|Average Per Epoch Time|Inference Time (Per Sample)|Average Per Step Time|Average Forward Pass Time|
|---|---|---|---|---|---|---|---|---|---|
|97.41%|3.99%|89.16%|7.337%|89.28%|7.61%|157.7 seconds|3 seconds (612 micro-seconds)|100.9 milli-seconds|797.64 micro-seconds|

The Confusion Matrix obtained after 100 epoch training of ResNet v2 with Mish on CIFAR-10 is shown below:

<div style="text-align:center"><img src ="Observations/confusion_100.PNG"  width="500"/></div>
<br>

The classification accuracies for the individual class labels are: 

|Class Labels| Classification Accuracy|
|---|---|
|Aeroplane|73.167%|
|Automobile|80.33%|
|Bird|70.33%|
|Cat|69.33%|
|Deer|74.33%|
|Dog|66.83%|
|Frog|75%|
|Horse|76.16%|
|Ship|77.667%|
|Truck|80.83%|

#### Wide Residual Networks (WRN):

Wide Residual Networks(WRN)([Paper](https://arxiv.org/abs/1605.07146)) of 4 variants were used for classification of CIFAR-10 dataset. The variants used and their corresponding evaluation metrics are observed below:

##### WRN 10-2:

|Activation Function |Accuracy|Loss|
|---|---|---|
|ReLU|84.52%|6.298%|
|Swish|86.56%|5.726%|
|Mish|**86.83%**|**5.712%**|

<em> *Number of Epochs=50, Batch Size= 128.
</em><br>

##### WRN 16-4:

|Activation Function |Accuracy|Loss|
|---|---|---|
|ReLU|**90.74%**|5.644%|
|Swish|90.07%|**5.014%**|
|Mish|90.54%|5.104%|

<em> *Number of Epochs=50, Batch Size= 128.
</em><br>

##### WRN 22-10:

|Activation Function |Accuracy|Loss|
|---|---|---|
|ReLU|**91.28%**|5.588%|
|Swish|90.17%|5.136%|
|Mish|90.38%|**4.961%**|

<em> *Number of Epochs=50, Batch Size= 128.
</em><br>

#### SimpleNet:

SimpleNet ([Paper](https://arxiv.org/abs/1608.06037)) was used for CIFAR-10 classification. The findings are provided in the table below:

|Activation Function |Accuracy|Loss| Top 3 Accuracy| Top 5 Accuracy|
|---|---|---|---|---|
|ReLU|91.16%|2.897%|98.62%|99.65%|
|Swish|91.44%|2.944%|**98.87%**|**99.77%**|
|Mish|**91.70%**|**2.759**|98.85%|99.75%|

<em> *Number of Epochs=50, Batch Size=128, Network Parameters= 5.59 M
</em><br>

### CIFAR-100:

CIFAR-100  ([Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)) is another subset of the Tiny Image Dataset similar to CIFAR-10, however containing 60000 images belonging to 100 classes with 600 images per class. All these images are 32x32 RGB images. These 100 classes are then grouped into 20 super-classes. The images are annotated with 2 labels- Fine Label (The class it belongs to) and Coarse Label (The super-class it belongs to). CIFAR-100 is also used extensively for Image Classification benchmarks, and here ResNet and WRN (Wide Residual Network) of various variants were used to benchmark Mish Activation Function against the likes of ReLU and Swish.

#### ResNet-56:

ResNet v2-56 was used for CIFAR-100 as similar to the CIFAR-10 classification. Here, however, Batch Size and Epochs were varied. Batch Size was varied to accelerate training and Epochs were varied to observe the changes in the evaluation metrics. All details have been provided subsequently. 

- For Batch Size=32:

|Activation Function |Accuracy (5*)|Loss (5*)|Accuracy (10*)|Loss (10*)|Accuracy (50*)|Loss(50*)
|---|---|---|---|---|---|---|
|ReLU|28.40%|36.91%|35.60%|32.99%|57.25%|22.9%|
|Swish|**31.28%**|**35.61%**|40.30%|30.93%|60.28%|22.06%|
|Mish|30.55%|36%|**41.11%**|**30.40%**|**60.67%**|**21.54%**|

<em> *This indicates the number of epochs
</em><br>

The Confusion Matrix obtained after 50 epoch training of ResNet v2 with Mish on CIFAR-100 is shown below:

<div style="text-align:center"><img src ="Observations/Confusion_50.png"  width="500"/></div>
<br>

#### NASNet:

<div style="text-align:center"><img src ="Observations/Nasnet_100.png"  width="1000"/></div>
<br>

#### Wide Residual Networks (WRN):

Wide Residual Networks(WRN) of 4 variants were used for classification of CIFAR-100 similar to the classification task of CIFAR-10 dataset. The variants used and their corresponding evaluation metrics are observed below:

##### WRN 10-2:

|Activation Function |Accuracy (Mean of 3 Runs)|
|---|---|
|ReLU|62.5567%|
|Swish|66.98%|
|Mish|**67.157%**|

<em> *Number of Epochs=125, Batch Size= 128.
</em><br>

##### WRN 16-4:

|Activation Function |Accuracy|
|---|---|
|ReLU|74.60%|
|Swish|74.60%|
|Mish|**74.92%**|

<em> *Number of Epochs=125, Batch Size= 128.
</em><br>

##### WRN 22-10:

|Activation Function |Accuracy|
|---|---|
|ReLU|72.2%|
|Swish|71.89%|
|Mish|**72.32%**|

<em> *Number of Epochs=50, Batch Size= 128.
</em><br>

##### WRN 40-4:

|Activation Function |Accuracy|
|---|---|
|ReLU|69.35%|
|Swish|**69.59%**|
|Mish|69.52%|

<em> *Number of Epochs=50, Batch Size= 128.
</em><br>

### Custom Data-Sets:

#### ASL (American Sign Language):

Custom CNN (Convolutional Neural Network) was used for classification ASL ([American Sign Language Dataset](https://www.kaggle.com/datamunge/sign-language-mnist)). The Evaluation Metrics table is given below:

| Activation Function  | Accuracy (10*) |  Loss (10*) |
| ------------- | ------------- | ---|
| ReLU  | 74.42%  |7.965%|
| Swish  | 68.84%  |10.464%|
| Mish  | **77.38%**|**7.078%**|

<em> *The number indicates the Number of Epochs
</em><br>

#### Malaria Cells Dataset:

Deep Conv Net was used for classifying microscopic cellular images of healthy cells and malaria paracitized cells present in the Malaria Cells Dataset ([Dataset](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)). The comparative analysis of the metrics scores obtained from the network using Mish against ReLU and Swish is given in the table below:

| Activation Function  | Accuracy (10*) |  Loss (10*) |
| ------------- | ------------- | ---|
| ReLU  | 94.21%  |**1.45%**|
| Swish  | **95.97%**  |**1.45%**|
| Mish  | 95.12%|1.56%|

<em> *The number indicates the Number of Epochs
</em><br>

#### Caravan Image Masking Challenge Dataset:

U-Net ([Paper](https://arxiv.org/abs/1505.04597)) was deployed for the Caravan Image Masking Challenge ([Challenge/Dataset](https://www.kaggle.com/c/carvana-image-masking-challenge)) on [Kaggle](https://www.kaggle.com/). The evaluation metrics are given below in the table where the Loss and Dice Losses are being compared: 

| Activation Function  | Training Loss (5*) |  Training Dice-Loss (5*) | Validation Loss(5*)| Validation Dice-Loss(5*)| Average Epoch Time | Average Step Time|
| ------------- | ------------- | ---|---|---|---|---|
| ReLU  |  0.724% |0.119%|0.578%|0.096%|**343.2 seconds**|**253 milli-seconds**|
| Swish  | 0.665%|0.111%|0.639%|0.108%|379 seconds|279.2 milli-seconds|
| Mish  |**0.574%**|**0.097%**|**0.554%**|**0.092%**|411.2 seconds|303 milli-seconds|

<em> *The number indicates the Number of Epochs
</em><br>

The following graph shows the Loss Plotting for U-Net with Mish: (Values Scaled to loss value/10)

<div style="text-align:center"><img src ="Observations/loss.PNG"  width="700"/></div>
<br>

Some Test Samples obtained from the network:

<div style="text-align:center"><img src ="Observations/tests.PNG"  width="400"/></div>
<br>

### Generative Models

#### Auto-Encoders: 

A simple Auto-Encoder model was used on Fashion MNIST data to observe the characteristic MSE loss trends of Mish as compared to ReLU and Swish. The Mean Squared Error (MSE) values recorded for 50 iterations are recorded in the table provided below: 

|Activation Function|MSE|
|---|---|
|ReLU|0.0053245881572|
|Swish|0.00525206327438|
|Mish|**0.005139515735**|

Some samples obtained:

<div style="text-align:center"><img src ="Observations/Test.PNG"  width="400"/></div>
<br>

### GAN:

Generative Adversarial Networks ([GAN](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)) was used on MNIST dataset and the efficiency of Mish was compared to ReLU and Swish. 

|Activation Function| Generator Loss| Discriminator Loss|
|---|---|---|
|ReLU|5.1214063%|**11.78977%**|
|Swish|**4.8570448%**|12.737954%|
|Mish|5.02091%|13.451806%|

Some samples generated over 100 epochs: 

<div style="text-align:center"><img src ="Observations/MNIST-GAN.PNG"  width="400"/></div>
<br>

 Discriminator and Generator Loss Curves: 
 
<div style="text-align:center"><img src ="Observations/Loss.png"  width="400"/></div>
<br>
