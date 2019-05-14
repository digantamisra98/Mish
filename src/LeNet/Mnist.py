#Import necessary packages
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import mnist

def load_mnist():
    # Download the MNIST Dataset
    print("Downloading MNIST")
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    # Reshape Data based on Channel Ordering of the Network
    if K.image_data_format() == "channels_first":
	       trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
	       testData = testData.reshape((testData.shape[0], 1, 28, 28))

    else:
	       trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
	       testData = testData.reshape((testData.shape[0], 28, 28, 1))

    #Data Scaling
    trainData = trainData.astype("float32") / 255.0
    testData = testData.astype("float32") / 255.0

    # Generating Label Vectors for each classes in the Dataset
    trainLabels = np_utils.to_categorical(trainLabels, 10)
    testLabels = np_utils.to_categorical(testLabels, 10)

    return trainData, testData, trainLabels, testLabels
