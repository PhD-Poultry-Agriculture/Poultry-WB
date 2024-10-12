#%% Exercise 3 - Neural Link
# German Shiklov
# 317634517
################
#%%
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from mlp_functions import *
import os

# %% Load images and labels
current_folder = '/Users/gshiklov/Documents/Projects/GitProjects/Jeremaiha/GG-Machine/NeuralLink/Exercise3'
path = os.path.join(current_folder, 'MNIST_data/')

ytest = loadMNISTLabels(path + r't10k-labels.idx1-ubyte')
ytrain = loadMNISTLabels(path + r'train-labels.idx1-ubyte')

Xtest_raw = loadMNISTImages(path + r't10k-images.idx3-ubyte')
Xtrain_raw = loadMNISTImages(path + r'train-images.idx3-ubyte')

print(np.shape(Xtrain_raw))

# %% display a random image with label:
# img_index = np.random.randint(np.size(Xtrain_raw, axis=0))
# img = Xtrain_raw[img_index, :, :]
# plt.imshow(img, cmap='gray')
# plt.title(str(ytrain[img_index]))
# plt.show()

# %% preprocess the images (reshape to vectors and subtract mean)
Xtrain = preprocess(Xtrain_raw)
Xtest = preprocess(Xtest_raw)

# %% Parameters
# The first and last values in layer_sizes should be equal to the input and
# output dimensions respectively. Try different values for the layer sizes
# inbetween and see how they affect the performance of the network.

layers_sizes = [784, 32, 10] # flexible, but must be [784,...,10]
epochs = 4      # number of times to repeat over the whole training set
eta = 0.1       # learning rate
batch_size = 32 # number of samples in each training batch

# %% Initialize weights
# The weights are initialized to normally distributed random values. Note
# that we scale them by the previous layer size so that the input to
# neurons in different layers will be of similar magnitude.

n_weights = len(layers_sizes)-1
weights = np.zeros((2,), dtype=object)
for i in range(n_weights):
    weights[i] = np.divide(np.random.standard_normal((layers_sizes[i+1],layers_sizes[i])), layers_sizes[i])

#%%
def select_uniform_subset(X, y, examples_per_class=1000):
    unique_classes = np.unique(y)
    indices = []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        cls_sample_indices = np.random.choice(cls_indices, examples_per_class, replace=False)
        indices.extend(cls_sample_indices)

    return X[indices, :, :], y[indices]

Xtrain_raw_subset, ytrain_subset = select_uniform_subset(Xtrain_raw, ytrain)
Xtrain_subset = preprocess(Xtrain_raw_subset)

# %% Training for a Subset - 1.2.3
N = np.size(Xtrain_subset, axis=0)  # Update to use the size of the subset

n_mbs = np.ceil(N / batch_size).astype(np.int16)

batch_loss = np.empty((epochs * n_mbs, )) * np.nan
test_loss = np.empty((epochs * n_mbs, )) * np.nan
test_acc = np.empty((epochs * n_mbs, )) * np.nan
iteration = 0

for i in range(epochs):
    perm = np.random.permutation(N)
    for j in range(n_mbs):
        idxs = perm[(batch_size * j):min((batch_size * (j + 1)), N)]
        X_mb = Xtrain_subset[idxs, :]
        y_mb = ytrain_subset[idxs]
        grads, loss = backprop(weights, X_mb, y_mb)
        batch_loss[iteration] = loss
        test_acc[iteration], test_loss[iteration] = test(weights, Xtest, ytest)
        for k in range(len(weights)):
            weights[k] = weights[k] - eta * grads[k]
        iteration += 1

    acc, loss = test(weights, Xtest, ytest)
    print(f'Done epoch {i}, test accuracy: {acc:.6f}\n')

#%% Training in General - 1.2.1/2/4
N = np.size(Xtrain, axis=0)
n_mbs = np.ceil(N/batch_size).astype(np.int16)

# create vectors to keep track of loss:
batch_loss = np.empty((epochs * n_mbs, )) * np.nan
test_loss = np.empty((epochs * n_mbs, )) * np.nan
test_acc = np.empty((epochs * n_mbs, )) * np.nan
iteration = 0
for i in range(epochs):
    perm = np.random.permutation(N)
    count = 0
    for j in range(n_mbs):
        idxs = perm[(batch_size * j):min((batch_size * (j+1))-1, N-1)]

        # pick a batch of samples:
        X_mb = Xtrain[idxs, :]
        y_mb = ytrain[idxs]

        # compute the gradients:
        grads, loss = backprop(weights, X_mb, y_mb)

        # keep track of the batch loss
        batch_loss[iteration] = loss

        # uncomment the next line to keep track of test loss and error.
        test_acc[iteration], test_loss[iteration] = test(weights,Xtest,ytest)
        # Note: evaluating the test_loss for each batch will slow down
        # computation. If it is too slow you can instead evaluate the test
        # loss at a lower frequency (once every 10 batches or so...)

        # update the weights:
        for k in range(len(weights)):
            weights[k] = weights[k] - eta * grads[k]

        iteration = iteration + 1  # counts the number of updates

    acc, loss = test(weights, Xtest, ytest)
    print('Done epoch %d, test accuracy: %f\n' % (i, acc))

# %% Plot some results - Initial simulations - 1.2.1 / 1.2.2
# Example plot of the learning curve
# fig, ax1 = plt.subplots()
# ax1.plot(batch_loss, 'r-', label='Training loss')
# ax1.plot(test_loss, 'k-', label='Test loss')
# ax1.set_xlabel('Iteration')
# ax1.set_ylabel('Loss')

# ax2 = ax1.twinx()
# ax2.plot(test_acc, label='Test accuracy')
# ax2.set_ylabel('Accuracy')
# fig.legend()
# plt.show()

#%%
# Normalize the losses
normalized_batch_loss = batch_loss / np.max(batch_loss)
normalized_test_loss = test_loss / np.max(test_loss)
test_error = 1 - test_acc   # Calculate test error (1 - accuracy)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(normalized_batch_loss, label='Training Loss')
plt.plot(normalized_test_loss, label='Test Loss')
plt.plot(test_error, label='Test Error')
plt.xlabel('Epochs')
plt.ylabel('Normalized Loss/Error')
plt.legend()
plt.title('Training Progress')
plt.show()
# %% Display 10 misclassifications with highest loss
yhat, output = predict(weights, Xtest)
t = np.zeros((len(ytest), 10))
for i in range(len(ytest)):
    t[i, ytest[i]] = 1 

test_losses = np.sum((output - t) ** 2, axis=1)
sorted_index = np.argsort(-test_losses)
idxs = sorted_index[:10]

plt.figure(figsize=(15, 6))
for k in range(10):
    ax = plt.subplot(2, 5, k + 1)
    idx = sorted_index[k]
    x = Xtest_raw[idx, :, :]
    ax.imshow(x, cmap='gray')
    ax.set_title(f'True: {ytest[idx]}, Pred: {yhat[idx]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# %% Display a sample of correct images
correct_indices = np.where(yhat == ytest)[0]
if len(correct_indices) >= 10:
    selected_correct_indices = correct_indices[:10]
else:
    selected_correct_indices = correct_indices

plt.figure(figsize=(15, 6))
for i, idx in enumerate(selected_correct_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(Xtest_raw[idx], cmap='gray')
    plt.title(f"True: {ytest[idx]}, Pred: {yhat[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()