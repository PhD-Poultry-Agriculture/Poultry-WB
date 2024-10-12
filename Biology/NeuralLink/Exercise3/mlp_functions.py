def backprop(weights, X, y):
    """
    This function receives a set of weights, a matrix with images
    and the corresponding labels. The output should be an array
    with the gradients of the loss with respect to the weights, averaged over
    the samples. It should also output the average loss of the samples.
    :param weights: an array of length L where the n-th cell contains the
    connectivity matrix between layer n-1 and layer n.
    :param X: samples matrix (match the dimensions to your input)
    :param y: corresponding labels
    :return:
    """    
    print(f"Backpropagation - X shape: {X.shape}, y shape: {y.shape}")
    y_one_hot = np.eye(10)[y]
    print(f"y_one_hot shape: {y_one_hot.shape}")
    activations, pre_activations = forward_pass(X, weights)
    grads = backward_pass(y_one_hot, activations, pre_activations, weights, X)
    mean_loss = np.mean(np.square(activations[-1] - y_one_hot))
    print(f"Mean loss: {mean_loss}")

    return grads, mean_loss



def test(weights, Xtest, ytest):
    """
    This function receives the Network weights, a matrix of samples and
    the corresponding labels, and outputs the classification
    accuracy and mean loss.
    The accuracy is equal to the ratio of correctly labeled images.
    The loss is given the square distance of the last layer activation
    and the 0-1 representation of the true label
    Note that ytest in the MNIST data is given as a vector of labels from 0-9. To calculate the loss you
    need to convert it to 0-1 (one-hot) representation with 1 at the position
    corresponding to the label and 0 everywhere else (label "2" maps to
    (0,0,1,0,0,0,0,0,0,0) etc.)
    :param weights: array of the network weights
    :param Xtest: samples matrix (match the dimensions to your input)
    :param ytest: corresponding labels
    :return:
    """

    print(f"Testing - Xtest shape: {Xtest.shape}, ytest shape: {ytest.shape}")
    predicted_labels, last_layer_activation = predict(weights, Xtest)
    y_one_hot = np.eye(10)[ytest]
    mean_loss = np.mean(np.square(last_layer_activation - y_one_hot))
    accuracy = np.mean(predicted_labels == ytest)
    print(f"Test accuracy: {accuracy}, Mean loss: {mean_loss}")
    
    return accuracy, mean_loss

def predict(weights, X):
    """
    The function takes as input an array of the weights and a matrix (X)
    with images. The outputs should be a vector of the predicted
    labels for each image, and a matrix whose columns are the activation of
    the last layer for each image.
    last_layer_activation should be of size [10 X num_samples]
    predicted_labels should be of size [1 X num_samples] or [10 X num_samples]
    The predicted label should correspond to the index with maximal
    activation in the last layer
    :param weights: array of the network weights
    :param X: samples matrix (match the dimensions to your input)
    :return:
    """
    print(f"Predict - X shape: {X.shape}")
    activations, _ = forward_pass(X, weights)
    last_layer_activation = activations[-1]
    predicted_labels = np.argmax(last_layer_activation, axis=1)
    print(f"Predicted labels shape: {predicted_labels.shape}, Last layer activation shape: {last_layer_activation.shape}")
    return predicted_labels, last_layer_activation

def backward_pass(y_true, activations, pre_activations, weights, X):
    """
    Performs the backward pass to compute gradients with respect to activations, pre-activations, and weights.
    
    :param y_true: True labels for the batch, one-hot encoded.
    :param activations: List of activations for each layer from forward pass.
    :param pre_activations: List of pre-activations for each layer from forward pass.
    :param weights: List of weight matrices for the network.
    :return: List of gradients with respect to weights.
    """
    print(f"Backward pass - y_true shape: {y_true.shape}")
    deltas = [loss_derivative(activations[-1], y_true)]
    grads = [None] * len(weights)
    for l in range(len(weights) - 1, -1, -1):
        if l == len(weights) - 1:
            grad = np.dot(deltas[-1].T, activations[l]) / y_true.shape[0]
        else:
            delta = np.dot(deltas[-1], weights[l + 1]) * relu_derivative(pre_activations[l])
            deltas.append(delta)
            grad = np.dot(delta.T, activations[l]) / y_true.shape[0]
        
        grads[l] = grad
    deltas = list(reversed(deltas))

    return grads

def forward_pass(X, weights):
    """
    Performs the forward pass through the network and stores pre-activations and activations for each layer.
    
    :param X: Input data of shape (num_samples, num_features), representing a batch of samples.
    :param weights: List of weight matrices, where weights[l] represents the weight matrix for layer l.
    :return: Tuple (activations, pre_activations), where each is a list containing the activations or pre-activations for each layer.
    """
    print(f"Forward pass - X shape: {X.shape}")
    activations = [X]
    pre_activations = []

    for W in weights:
        Z = np.dot(activations[-1], W.T)
        pre_activations.append(Z)
        A = relu_activation(Z)
        activations.append(A)
        print(f"Weight shape: {W.shape}, Z shape: {Z.shape}, A shape: {A.shape}")

    return activations, pre_activations

import numpy as np
def relu_activation(z):
    return np.maximum(0, z)

def loss_derivative(output_activations, y_true):
    return (output_activations - y_true) / y_true.shape[0]

def relu_derivative(z):
    return np.where(z > 0, 1, 0)
