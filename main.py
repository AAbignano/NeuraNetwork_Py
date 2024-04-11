from sklearn.datasets import fetch_openml

import numpy as np

from sklearn.model_selection import train_test_split

import time

def to_categorical(y):
    y = np.array(y, dtype="int")
    input_shape = y.shape

    y = y.ravel()

    num_classes = np.max(y) + 1
    n = y.shape[0]

    categorical = np.zeros((n, num_classes), dtype="float32")
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical



def relu(x, derivative=False):
    if derivative:
        return (x > 0) * 1
    return np.maximum(0.0, x)

def softmax(x, derivative=False):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)


def forward_pass(params, x_train):
    # input layer activations becomes sample
    params['A0'] = x_train

    # input layer to hidden layer 1
    params['Z1'] = np.dot(params['W1'], params['A0'])
    params['A1'] = relu(params['Z1'])

    # hidden layer 1 to hidden layer 2
    params['Z2'] = np.dot(params['W2'], params['A1'])
    params['A2'] = relu(params['Z2'])

    # hidden layer 2 to output layer
    params['Z3'] = np.dot(params['W3'], params['A2'])
    params['A3'] = softmax(params['Z3'])

    return params['A3']

def backward_pass(params, y_train, output):
    '''
        This is the backpropagation algorithm, for calculating the updates
        of the neural network's parameters.

        Note: There is a stability issue that causes warnings. This is 
              caused  by the dot and multiply operations on the huge arrays.
              
              RuntimeWarning: invalid value encountered in true_divide
              RuntimeWarning: overflow encountered in exp
              RuntimeWarning: overflow encountered in square
    '''
    change_w = {}

    # Calculate W3 update
    error = 2 * (output - y_train) / output.shape[0] * softmax(params['Z3'], derivative=True)
    change_w['W3'] = np.outer(error, params['A2'])

    # Calculate W2 update
    error = np.dot(params['W3'].T, error) * relu(params['Z2'], derivative=True)
    change_w['W2'] = np.outer(error, params['A1'])

    # Calculate W1 update
    error = np.dot(params['W2'].T, error) * relu(params['Z1'], derivative=True)
    change_w['W1'] = np.outer(error, params['A0'])

    return change_w

def update_network_parameters(params, learning_rate, changes_to_w):
    '''
        Update network parameters according to update rule from
        Stochastic Gradient Descent.

        θ = θ - η * ∇J(x, y), 
            theta θ:            a network parameter (e.g. a weight w)
            eta η:              the learning rate
            gradient ∇J(x, y):  the gradient of the objective function,
                                i.e. the change for a specific theta θ
    '''
    
    for key, value in changes_to_w.items():
        params[key] -= learning_rate * value



# Fetch dataset
x, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
x = (x/255).astype('float32')
y = to_categorical(y)


# Split dataset
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)


# Create neural network
learning_rate = 0.001

# number of nodes in each layer
input_layer   = 784
hidden_layer1 = 128
hidden_layer2 = 64
output_layer  = 10

params = {
    'W1':np.random.randn(hidden_layer1, input_layer)   * np.sqrt(1. / hidden_layer1),
    'W2':np.random.randn(hidden_layer2, hidden_layer1) * np.sqrt(1. / hidden_layer2),
    'W3':np.random.randn(output_layer, hidden_layer2)  * np.sqrt(1. / output_layer)
}

# Train Neural network
start_time = time.time()

num_epochs = 10
for epoch in range(num_epochs):

    # Train
    for x, y in zip(x_train, y_train):
        output = forward_pass(params, x)
        changes_to_w = backward_pass(params, y, output)
        update_network_parameters(params, learning_rate, changes_to_w)

    # Validation
    correct_predictions = 0
    total_samples = 0

    for x, y in zip(x_val, y_val):
        output = forward_pass(params, x)
        prediction = np.argmax(output)

        total_samples += 1
        correct_predictions += (prediction == np.argmax(y))
    
    # Calculate validation accuracy
    val_accuracy = correct_predictions / total_samples
    
    print(f'Epoch: [{epoch+1}/{num_epochs}], Time Spent: {time.time()-start_time:.2f}s, Accuracy: {val_accuracy:.2f}')
