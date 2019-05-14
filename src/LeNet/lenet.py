# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import keras
import functools
from keras.optimizers import SGD
import subprocess
import timeit
import sys
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
install('GPUtil')
install('psutil')
install('humanize')
import GPUtil
import humanize,psutil

#Defining Top-K Accuracy Metric
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'

#Defining Google Le-Net Architecture
def LeNet(inputShape,numClasses,act, lRate):
    model = Sequential()
    model.add(Conv2D(20, 5, padding="same",input_shape=inputShape, activation = act))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, 5, padding="same",activation = act ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation = act))


    model.add(Dense(numClasses))

    model.add(Activation("softmax"))
    model.summary()
    return model

##GPU-Information
#!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
#print(subprocess.getoutput('nvidia-smi'))

def eval_model(model,epochs,batch_size,trainData,trainLabels,testData,testLabels):
    print("Compiling Model")
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy", 'top_k_categorical_accuracy',top3_acc])
    ##GPU-Infostats
    subprocess.Popen("nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1 | sed s/%//g > ./GPU-stats.log",shell=True)
    start_time = timeit.default_timer()  ##Obtaining Training Time
    #Model Training
    print("Started Training")
    model.fit(trainData, trainLabels, batch_size, epochs,verbose=1)
    elapsed = timeit.default_timer() - start_time
    print("Total Training Time  (in seconds)",elapsed)
	#Testing the Model
    print("Testing")
    start_time = timeit.default_timer()  ##Obtaining Inference Time
    model.evaluate(testData, testLabels,batch_size, verbose=1)
    elapsed = timeit.default_timer() - start_time
    print("Inference Time (in seconds):",elapsed)


    # Computational Cost Function
    def mem_report():
        print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

        GPUs = GPUtil.getGPUs()
        for i, gpu in enumerate(GPUs):
            print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

    # Generate Report
    mem_report()
