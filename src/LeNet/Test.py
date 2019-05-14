import Mnist
import Mish
import Swish
import lenet

##Obtaining the Data(if using MNIST):
trainData,testData,trainLabels,testLabels=Mnist.load_mnist()

##Obtaining the Le-Net model
model=lenet.LeNet((28,28,1),10,Mish.mish_layer,0.01)

##Train and Test the model
lenet.eval_model(model,2,128,trainData,trainLabels,testData,testLabels)
