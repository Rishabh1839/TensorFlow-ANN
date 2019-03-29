from __future__ import absolute_import, division, print_function

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
# the testing part is where eour model is in the process of being tested in the test image and label arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
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

className = ['T-shirt', 'Trousers', 'Trunks', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
             'Bag', 'Boots']
# Before we train our model, we explore the dataset's format
# TensorFlow's datasets include 60,000 images reprsented as 28 x 28 pixels
train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

# Before training the network our data must be preprocessed
# The image of the boot in the training set has a pixel value in the range of 0 - 255
# the pyplot library helps us scale the values from 0 - 1
# feeding the neural network model the data
# values / 255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# values / 255
train_images = train_images / 255.0
test_images = test_images / 255.0

# Here the first 25 images from the training set and displaying the class name under images
# verifies data in correct format
# ready to train and build network
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(className[train_labels[i]])
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
    # This last layer is a 10 node softmax layer that returns an array of 10 probability scors that sum to 1
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
# The train_images and train_labels are the training arrays
# The model then learns to associate with the images and it's labels
# we ask the model to make predictions about a test set (test_images, test_labels)
# we call out the model.fit method to start the training
model.fit(train_images, train_labels, epochs=5)

# Accuracy evaluation
# We compare how the model performs on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)

# Making predictions with the trained model
# we can use it to make predictions about the images
predictions = model.predict(test_images)
# Here the model has predicted the label
predictions[0]
# A prediction is an array of 10 numbers that describe as the "confidence" of the model
# The image corresponds to each of the 10 different articles of clothing
np.argmax(predictions[0])
# Checking if the test label is correct or not
# our prediction would be an ankle boot
test_labels[0]


# now we can graph this to look at the full set of 10 channels
def plot_image(i, predictions_array, true_label, image):
    predictions_array, true_label, image = predictions_array[i], true_label[i], image[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(className[predicted_label],
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
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
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
    plot_image(i,predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()

# we finally use the trained model to make the predictions for a single image
# grabbing an image from the test data set
img = test_images[0]
print(img.shape)
# the tf.keras models are optimized to make predictions on a batch or either a collection
# Adding image to a batch wherer it's only a member
img = (np.expand_dims(img,0))
print(img.shape)
# predicting images
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), className, rotation=45)


