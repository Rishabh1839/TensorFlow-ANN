Description of the Project (Basic Classification of images using TensorFlow)
-----------------------------
In this program I created a basic classification neural network with the help of TensorFlow, Numpy & Matplotlib.
TensorFlow is Google's high computation Library used for Machine learning applications such as Neural Networks.
Numpy is another library for Python that supports multidimensional arrays and matrices. It also has a large collection of 
High level mathematical functions that helps operating these arrays. Our other Library Matplotlib is a numerical extension for Numpy. 
It provides OOP API for embedding, plots into applications using general purpose User Interface tool kits. OOP means Object Oriented Programming.
Now we are familiar with our libraries, lets get to what data sets we have used for this program.
TensorFlow provides us with an MNIST fashion data set that includes 60,000 images with 28 x 28 pixels.
The MNIST data set stands for Modified National Institute of Standards and Technology database.
Now we are familiar with what data set we use. Now let's get to the part of creating our Neural Network.
Fun Fact: MNIST data set is also known as the Hello World of Machine Learning programs for computer vision,
Now, We create our neural network by creating arrays used for training our images and labels.
They are labeled as (trainImages, trainLabels), (testImages, testLabels) The images are a 28 by 28 array with pixels that range from 0 to 255, The labels are ranged from 0 to 10.
I will be showing you what label belongs to what class below!"

("label ----> Class")

("0 ----> T shirt")

("1 ----> Trousers")

("2 ----> Pullover trunks")

("3 ----> Dress")

("4 ----> Coat")

("5 ---->  Sandal")

("6 ----> Shirt")

("7 ----> Sneaker")

("8 ----> Bag")

("9 ----> Boots")

Now before we train our model, we shall explore the dataset's format. TensorFlow's data sets include 60,000 images represented as 28 x 28 pixels as I mentioned earlier.
We won't be using all 60,000 images, instead we will be calling out certain images for testing. Before we start training the network, our data must be preprocessed.
We select an image of a boot in the training set that has a pixel value in the range of 0 - 255. Our pyplot library helps us scale the image's values from 0 - 1
We then feel our neural network model, the data that needs to be read. We will be displaying our first 25 images from the training set and the class name under the images.
We then verify the data in its correct format and finally we will be ready to train and build our network. Our next step is to create our model where the Neural Network requires layers of the model.
For creating this model we use TensorFlow's Keras model. These layers can extract representations from the data that is being fed to them.
These layers also have parameters that are learned during the process of its training. We use 3 layers, a Flatten and 2 Dense layers.
The first layer transforms the format of the images from a 2d array (28 x 28 = 784 pixels).
The other dense layers are densely or fully connected Neural layers.
The second layer has about 128 nodes that are also considered as Neurons.
The last layer is a 10 node softmax layer that returns an array of 10 probability scores that sum to 1.
Each node includes a score, it indicates the probability that the current image belongs to one of the 10 classes. Now that we have our layers, Our next step is to compile our model.
This step includes having to feed our model the training data. Those are the trainImages and trainLabels arrays.
Our model then learns how to associate with the images and it's labels. We ask our model to make predictions about a test set which are the following arrays (testImages, testLabels).
We then call our the model.fit method to start the training process. Next step involves an accuracy evaluation where we compare how the model performs on the test dataset.
We call out our predict method for making predictions with our trained model. We can use it to make predictions about the images.
A prediction is an array of 10 numbers that describes the confidence our our model's predictions.
The image corresponds to each of the 10 different articles of clothing that we labeled earlier. We have to check if our test label is correct or not.
he prediction would be expected to be a boot and a couple other labels that I have use to call out the images.
We use the pyplot library that helps us graph to look at the full set of 10 channels. We plot the first X test images, their predicted and the true label that belongs to it.
The color correct predictions will be in blue. The incorrect predictions will be in red.
We finally have to use the trained model to make predictions for just a single image by grabbing an image.
The keras models are optimized to make predictions on a batch or either a collection. We add the image to a batch where it's only considered as a member.


