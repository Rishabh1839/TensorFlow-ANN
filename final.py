from __future__ import absolute_import, division, print_function
# importing TensorFlow our main library for creating this Neural Network
# adding keras for our model layers
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# With the help of TensorFlow we will be using basic classification by importing the
# MNIST data set which is often used as the "Hello World" of Machine Learning programs for Computer Vision
# For this classification I will be using the Fashion mnist data set which contains images of handwritten digits
# in an identical format to the articles of clothing that we will be using
# importing the Fashion MNIST data set as fashion_mnist
fashion_mnist = keras.datasets.fashion_mnist
# here we train our images and labels along with testing them by using an array
# loading up the fashion mnist data set
# the training part allows the data that the model uses to learn from
# the testing part is where our model is in the process of being tested in the test image and label arrays
(trainImages, trainLabels), (testImages, testLabels) = fashion_mnist.load_data()
# The images are 28 by 28 numpy arrays with pixels that range from a 0 to 255
# the labels are an array of integers that range from a 0 to 10
# I will be showing what label belongs to what class

# label ----> Class
# 0 ----> T shirt
# 1 ----> Trousers
# 2 ----> Pullover trunks
# 3 ----> Dress
# 4 ----> Coat
# 5 ---->  Sandal
# 6 ----> Shirt
# 7 ----> Sneaker
# 8 ----> Bag
# 9 ----> Boots

className = ['T-shirt', 'Trousers', 'Trunks', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boots']
# Before we train our model, we explore the dataset's format
# TensorFlow's datasets include 60,000 images represented as 28 x 28 pixels
trainImages.shape
len(trainLabels)
trainLabels
testImages.shape
len(testLabels)

# Before training the network our data must be preprocessed
# The image of the boot in the training set has a pixel value in the range of 0 - 255
# the pyplot library helps us scale the values from 0 - 1
# feeding the neural network model the data
# values / 255
plt.figure()
plt.imshow(trainImages[0])
plt.colorbar()
plt.grid(False)
plt.show()

# values / 255
trainImages = trainImages / 255.0
testImages = testImages / 255.0

# Here the first 25 images from the training set and displaying the class name under images
# verifies data in correct format
# ready to train and build network
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap=plt.cm.binary)
    plt.xlabel(className[trainLabels[i]])
plt.show()

# The model is being constructed here where the Neural Network requires the layers of the model
# Then it compiles the model
# Setting up layers with the help of Keras layers
# These layers can extract representations from the data that's being fed to them
# these layers have parameters that are learned during the process of training
model = keras.Sequential([
    # This first layer transforms the format of the images from a 2d-Array (28 x 28 pixels)
    # This would be equal to a 784 pixels
    keras.layers.Flatten(input_shape=(28, 28)),
    # These two Dense layers are densely or fully connected neural layers
    # The first Dense layer has about a 128 nodes which are also known as neurons
    keras.layers.Dense(128, activation=tf.nn.relu),
    # This last layer is a 10 node softmax layer that returns an array of 10 probability scores that sum to 1
    # Each node has a score that indicates the probability that the current image belongs to one of the 10 classes
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Here we compile our model
# the optimizer is how the model is updated which is based on the data that it notices and its loss function
model.compile(optimizer='adam',
              # This loss function measures the accuracy of the model during the training process
              loss='sparse_categorical_crossentropy',
              # metrics is used to monitor the training and testing steps
              metrics=['accuracy'])

# Training our model
# There are few steps that are followed for training the data to our model
# We feed the training data to our model that we have created
# The trainImages and trainLabels are the training arrays
# The model then learns to associate with the images and it's labels
# we ask the model to make predictions about a test set (testImages, testLabels)
# we call out the model.fit method to start the training
model.fit(trainImages, trainLabels, epochs=5)

# Accuracy evaluation
# We compare how the model performs on the test dataset
testLoss, testAccuracy = model.evaluate(testImages, testLabels)
print('Test accuracy: ', testAccuracy)

# Making predictions with the trained model
# we can use it to make predictions about the images
predictions = model.predict(testImages)
# Here the model has predicted the label
predictions[0]
# A prediction is an array of 10 numbers that describe as the "confidence" of the model
# The image corresponds to each of the 10 different articles of clothing
np.argmax(predictions[0])
# Checking if the test label is correct or not
# our prediction would be an ankle boot
testLabels[0]


# now we can graph this to look at the full set of 10 channels
def plot_image(i, predictions_array, true_label, image):
    predictions_array, true_label, image = predictions_array[i], true_label[i], image[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image, cmap=plt.cm.binary)

    labelPredicted = np.argmax(predictions_array)
    if labelPredicted == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(className[labelPredicted],
                                         100 * np.max(predictions_array),
                                         className[true_label]), color=color)


def plot_value_array(i, predicitons_array, true_label):
    predicitons_array, true_label = predicitons_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predicitons_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predicitons_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('cyan')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, testLabels, testImages)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, testLabels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, testLabels, testImages)
plt.subplot(1,2,2)
plot_value_array(i, predictions, testLabels)
plt.show()

# plotting the first X test images, their predicted and true label
# The color correct predictions are in blue
# The incorrect predictions are in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, testLabels, testImages)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, testLabels)
plt.show()

# we finally use the trained model to make the predictions for a single image
# grabbing an image from the test data set
img = testImages[0]
print(img.shape)
# the tf.keras models are optimized to make predictions on a batch or either a collection
# Adding image to a batch wherer it's only a member
img = (np.expand_dims(img,0))
print(img.shape)
# predicting images
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, testLabels)
_ = plt.xticks(range(10), className, rotation=45)

print("=" * 100)
print("TensorFlow's Neural Network for Basic Classification.")
print("What does this program do?")
print("In this program I created a basic classification neural network with the help of TensorFlow, Numpy & Matplotlib.")
print("TensorFlow is Google's high computation Library used for Machine learning applications such as Neural Networks.")
print("Numpy is another library for Python that supports large multidimensional arrays and matrices.")
print("It also has a large collection of High level mathematical functions that helps operating these arrays.")
print("Our other Library Matplotlib is a numerical extension for Numpy. It provides OOP API for embedding,")
print("plots into applications using general purpose User Interface tool kits. OOP means Object Oriented Programming.")
print("Now we are familiar with our libraries, lets get to what data sets we have used for this program.")
print("TensorFlow provides us with an MNIST fashion data set that includes 60,000 images with 28 x 28 pixels.")
print("The MNIST data set stands for Modified National Institute of Standards and Technology database.")
print("Now we are familiar with what data set we use. Now let's get to the part of creating our Neural Network.")
print("Fun Fact: MNIST data set is also known as the Hello World of Machine Learning programs for computer vision,")
print("Now, We create our neural network by creating arrays used for training our images and labels.")
print("They are labeled as (trainImages, trainLabels), (testImages, testLabels)")
print("The images are a 28 by 28 array with pixels that range from 0 to 255, The labels are ranged from 0 to 10.")
print("I will be showing you what label belongs to what class below!")
print("=" * 100)
print("label ----> Class")
print("0 ----> T shirt")
print("1 ----> Trousers")
print("2 ----> Pullover trunks")
print("3 ----> Dress")
print("4 ----> Coat")
print("5 ---->  Sandal")
print("6 ----> Shirt")
print("7 ----> Sneaker")
print("8 ----> Bag")
print("9 ----> Boots")
print("=" * 100)
print("Now before we train our model, we shall explore the dataset's format.")
print("TensorFlow's data sets include 60,000 images represented as 28 x 28 pixels as I mentioned earlier.")
print("We won't be using all 60,000 images, instead we will be calling out certain images for testing.")
print("Before we start training the network, our data must be preprocessed.")
print("We select an image of a boot in the training set that has a pixel value in the range of 0 - 255.")
print("Our pyplot library helps us scale the image's values from 0 - 1")
print("We then feel our neural network model, the data that needs to be read.")
print("We will be displaying our first 25 images from the training set and the class name under the images.")
print("We then verify the data in its correct format and finally we will be ready to train and build our network.")
print("Our next step is to create our model where the Neural Network requires layers of the model.")
print("For creating this model we use TensorFlow's Keras model.")
print("These layers can extract representations from the data that is being fed to them.")
print("These layers also have parameters that are learned during the process of its training.")
print("We use 3 layers, a Flatten and 2 Dense layers.")
print("The first layer transforms the format of the images from a 2d array (28 x 28 = 784 pixels)")
print("The other dense layers are densely or fully connected Neural layers.")
print("The second layer has about 128 nodes that are also considered as Neurons.")
print("The last layer is a 10 node softmax layer that returns an array of 10 probability scores that sum to 1.")
print("Each node includes a score, it indicates the probability that the current image belongs to one of the 10 classes")
print("Now that we have our layers, Our next step is to compile our model.")
print("This step includes having to feed our model the training data.")
print("Those are the trainImages and trainLabels arrays.")
print("Our model then learns how to associate with the images and it's labels.")
print("We ask our model to make predictions about a test set which are the following arrays (testImages, testLabels)")
print("We then call our the model.fit method to start the training process.")
print("Next step involves an accuracy evaluation where we compare how the model performs on the test dataset.")
print("We call out our predict method for making predictions with our trained model.")
print("We can use it to make predictions about the images.")
print("A prediction is an array of 10 numbers that describes the confidence our our model's predictions.")
print("The image corresponds to each of the 10 different articles of clothing that we labeled earlier.")
print("We have to check if our test label is correct or not.")
print("The prediction would be expected to be a boot.")
print("We use the pyplot library that helps us graph to look at the full set of 10 channels.")
print("We plot the first X test images, their predicted and the true label that belongs to it.")
print("The color correct predictions will be in blue.")
print("The incorrect predictions will be in red.")
print("We finally have to use the trained model to make predictions for just a single image by grabbing an image.")
print("The keras models are optimized to make predictions on a batch or either a collection.")
print("We add the image to a batch where it's only considered as a member.")
