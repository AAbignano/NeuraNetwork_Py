# Neural Network from Scratch

This project was created as an exercise to learn and demonstrate my knowledge of the neural network by writing one in Python from scratch including the activation function (relu and softmax), forward pass and backpropagation.

The neural network is trained on a subset of the MNIST dataset consisting of 28x28 images of handwritten digits. This code will download that dataset and split it into 85/15 for training and validation.

The neural network consists of 986 nodes across four layers: 784 nodes for the input layer, 128 nodes for the first hidden layer and 64 for the second, and 10 nodes for the output layer.

The initial model weights are sampled from a standard normal distribution and are trained over 10 epochs which should allow for an accuracy of 90%+


## Requirements to run
This project was built with Python 3.11 along with the following packages:  
numpy-1.26.4  
scikit-learn-1.4.1.post1
