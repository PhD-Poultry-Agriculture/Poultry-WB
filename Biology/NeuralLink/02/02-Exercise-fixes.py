# Author: German Shiklov
# ID: 317634517

#%% Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import norm
from copy import deepcopy
from pylab import scatter, show

#%% Load up
file_path = '/Users/gshiklov/Documents/Projects/GitProjects/Jeremaiha/GG-Machine/NeuralLink/02/data_10D.mat'
M = loadmat(file_path)
# %% Perceptron and Adatron Definitions
def perceptron(X, y0):
    t_max = 100
    N, P = X.shape
    w = np.zeros(N)
    did_converge = False

    for epochs in range(1, t_max + 1):
        w_old = np.copy(w)

        for i in range(P):
            if y0[i] * np.dot(w, X[:, i]) <= 0:
                w = w + y0[i] * X[:, i]

        if np.array_equal(w, w_old):
            did_converge = True
            break

    return w, did_converge, epochs

def adatron(K,y0, eps=0.01, t_max=2000):
    '''
    The AdaTron algorithm for SVM learning
    
    Parameters
    ----------
    K : PxP REAL MATRIX, where P is the training set size
        Contains all pairwise overlaps between training examples: X_ij= x_i*x_j
    y0 : Px1 REAL VECTOR
        All training labels, y_i = label(x_i)
    eps: FLOAT, optional
        Stopping criterion (when update<eps). The default is .0001
    t_max : INT, optional
        max runtime in epochs. The default is 1000.

    Returns
    -------
    hasConverged : BOOLEAN
        whether the algorithm converged (no more updates) or reached t_max and eas stopped.
    A : ExP REAL MATRIX, where E is the number of epochs
        Contains the P-dim support coefficient vectors (alpha) for all the epochs.
        A[t,:] is the support vector alpha_t (at epoch=t)
        When the algorithm has converges, A[-1,:] is the final result that defines the decision rule 

    '''
    print("Training Adatron...")

    P, P1 = K.shape
    assert P == P1, "Kernel matrix K should be PxP, where P is the training set size"

    # Print shapes for debugging
    print(f"Shape of K: {K.shape}")
    print(f"Shape of y0: {y0.shape}")

    assert y0.size == P, "Input-output set size mismatch"

    hasConverged = False
    epochs = 0
    A = []
    eta = 0.2/(np.max(np.linalg.norm(K,axis=0)))
    A.append(np.zeros((P,1)))
    
    while ((not hasConverged) and (epochs < t_max)):
        a = deepcopy(A[-1])
        for mu in range(P):
            y = y0[mu]
            coeff = y*(a*y0).T
            da = max(-a[mu],eta*(1-coeff@K[:,mu]))
            if np.isinf(da):
                print("stopping because of exploding updates, epoch={}, mu={}".format(epochs,mu))
                hasConverged = True
                continue;
            if np.isnan(da):
                print("nan")
                hasConverged = True
                continue;
            a[mu] += da
        A.append(a)
        diff = abs(A[-1]-A[-2])
        update_flag = np.max(diff) > eps
        hasConverged = hasConverged or (not update_flag)
        epochs += 1
        
    return hasConverged, np.squeeze(np.array(A))

# %% 1.2: Linear separation
X = M['X'].T
y = M['y0'].flatten()
true_weight_vector = M['w0'].flatten()
cos_distances = []
P_values = [10, 50, 100, 300]
num_trials = 10
generalization_performance_adatron = []
generalization_performance_perceptron = []
avg_support_vectors = []

print(X.shape)
#%%

def select_training_set(X, y, P):
    indices = np.random.permutation(X.shape[0])[:P]
    X_train = X[indices, :]
    y_train = y[indices]
    return X_train, y_train, indices

def calculate_kernel_matrix(X_train):
    return X_train @ X_train.T

def train_and_evaluate_model(X_train, y_train, X_test, y_test, true_weight_vector):
    y_train_reshaped = y_train.reshape(-1, 1)
    K = calculate_kernel_matrix(X_train)
    assert K.shape[0] == K.shape[1] == y_train_reshaped.shape[0], "Shape mismatch among K and y_train_reshaped"
    
    # AdaTron algorithm application
    has_converged, A = adatron(K, y_train_reshaped)
    alpha = A[-1].flatten()
    
    alpha_reshaped = alpha[:, np.newaxis]
    model_weight_vector = np.sum(alpha_reshaped * X_train, axis=0)
    assert model_weight_vector.shape[0] == X_train.shape[1], "Model weight vector shape mismatch"
    
    # Perceptron algorithm application
    w_perceptron, did_converge, epochs = perceptron(X_train.T, y_train)
    assert w_perceptron.shape[0] == X_train.shape[1], "Perceptron weight vector shape mismatch"
    
    # Calculate cosine similarity and distance
    cosine_distance = calculate_cosine_distance(model_weight_vector, true_weight_vector)
    
    # Count correct predictions by Perceptron
    correct_perceptron = sum(np.sign(np.dot(w_perceptron, X_test[i, :])) == y_test[i] for i in range(len(y_test)))
    
    return {
        "cosine_distance": cosine_distance,
        "correct_perceptron": correct_perceptron,
        "alpha": alpha,
        "w_perceptron": w_perceptron
    }

def calculate_cosine_distance(model_weight_vector, true_weight_vector):
    model_normed = model_weight_vector / np.linalg.norm(model_weight_vector)
    true_normed = true_weight_vector / np.linalg.norm(true_weight_vector)
    cosine_similarity = np.dot(model_normed, true_normed)
    return 1 - abs(cosine_similarity)

#%%

def run_simulation(X, y, P_values, num_trials, true_weight_vector):
    generalization_performance_adatron = []
    generalization_performance_perceptron = []
    avg_support_vectors = []
    cos_distances = []

    
    for P in P_values:
        correct_perceptron = 0
        support_vector_sum = 0
        cos_distances_for_P = []
        support_vectors_for_P = []  # Collecting support vector counts for each trial

        for trial in range(num_trials):
            P = min(P, X.shape[0] - 1)
            X_train, y_train, indices = select_training_set(X, y, P)
            X_test, y_test = np.delete(X, indices, axis=0), np.delete(y, indices, axis=0)

            model_metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, true_weight_vector)
            cos_distances_for_P.append(model_metrics["cosine_distance"])
            correct_perceptron += model_metrics["correct_perceptron"]
            sv_count = np.count_nonzero(model_metrics["alpha"] > 0)
            support_vector_sum += sv_count
            support_vectors_for_P.append(sv_count)  # Append support vector count for this trial

        # Calculate averages and performance metrics
        avg_support_vectors.append(support_vector_sum / num_trials)
        cos_distances.append(np.mean(cos_distances_for_P))
        generalization_performance_perceptron.append(correct_perceptron / ((X.shape[0] - P) * num_trials))

        # Plot results for the current P after each P's trials are finished
        plt.figure(figsize=(15, 5))

        # Plotting cosine distances
        plt.subplot(1, 3, 1)
        plt.plot(range(1, num_trials + 1), cos_distances_for_P, marker='o', label=f'P={P}')
        plt.xlabel('Trial')
        plt.ylabel('Cosine Distance')
        plt.title('Cosine Distance per Trial')
        plt.legend()

        # Plotting average number of support vectors
        plt.subplot(1, 3, 2)
        plt.plot(range(1, num_trials + 1), support_vectors_for_P, marker='x', label=f'P={P}')
        plt.xlabel('Trial')
        plt.ylabel('Number of Support Vectors')
        plt.title('Support Vectors per Trial')
        plt.legend()

        # Plotting generalization performance for Perceptron
        plt.subplot(1, 3, 3)
        generalization_performance = [model_metrics["correct_perceptron"] / (len(y_test)) for _ in range(num_trials)]
        plt.plot(range(1, num_trials + 1), generalization_performance, marker='^', label=f'P={P}')
        plt.xlabel('Trial')
        plt.ylabel('Generalization Performance (Perceptron)')
        plt.title('Generalization Performance per Trial')
        plt.legend()

        plt.tight_layout()
        plt.show()


    return {
        "generalization_performance_perceptron": generalization_performance_perceptron,
        "avg_support_vectors": avg_support_vectors,
        "cos_distances": cos_distances
    }

# Example simulation call (assuming necessary functions and variables are defined)
print(run_simulation(X, y, P_values, num_trials, true_weight_vector))



#%% 1.3 Nonlinear separation using the kernel trick - Part 1
def kernel_polynomial(X, d):
    print(f"Polynomial Kernel: Computing with degree {d} on input of shape {X.shape}")
    K = (X @ X.T + 1) ** d
    print(f"Polynomial Kernel: Resulting kernel matrix shape {K.shape}")
    return K

def kernel_gaussian(X, sigma):
    print(f"Gaussian Kernel: Computing with sigma {sigma} on input of shape {X.shape}")
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * X @ X.T
    K = np.exp(-sq_dists / (2 * sigma**2))
    print(f"Gaussian Kernel: Resulting kernel matrix shape {K.shape}")
    return K

def checkboard(N):
    print(f"Generating checkerboard pattern with {N} points.")
    X=np.random.rand(2,N)
    p=np.mod(np.ceil(X*3),2)
    y0=2.*np.logical_xor(p[0,:],p[1,:])-1.
    return X,y0

def test_checkboard():
    print("Testing checkerboard pattern.")
    x,y=checkboard(2000)
    scatter(x[0,y==-1.],x[1,y==-1.],c='r')
    scatter(x[0,y==1.],x[1,y==1.],c='b')

def train_adatron(X, y, kernel_func, **kwargs):
    print(f"Training with {kernel_func.__name__} kernel...")
    K = kernel_func(X, **kwargs)
    print(f"Kernel matrix shape: {K.shape}")
    y_reshaped = y.reshape(-1, 1)
    has_converged, A = adatron(K, y_reshaped)
    print(f"Training completed with convergence status: {has_converged}")
    print(f"Shape of A from adatron: {A.shape}")
    return A

def predict_adatron(X_train, X_test, A, y_train, kernel_func, **kwargs):
    print(f"Predicting with {kernel_func.__name__} kernel...")
    A_last_epoch = A[-1, :] if A.ndim > 1 else A
    A_flat = A_last_epoch.flatten()
    print(f"Shape of A_last_epoch: {A_last_epoch.shape}, Shape of A_flat: {A_flat.shape}")
    y_train_flat = y_train.flatten() if y_train.ndim > 1 else y_train
    print(f"Shape of y_train_flat: {y_train_flat.shape}")

    if kernel_func.__name__ == "kernel_polynomial":
        K_test = kernel_func(np.vstack([X_train, X_test]), kwargs['d'])
        K_test = K_test[:len(X_train), len(X_train):]
    elif kernel_func.__name__ == "kernel_gaussian":
        K_test = kernel_func(np.vstack([X_train, X_test]), sigma=kwargs['sigma'])
        K_test = K_test[:len(X_train), len(X_train):]
    else:
        raise ValueError("Unsupported kernel function.")

    print(f"Shape of K_test: {K_test.shape}")
    predictions_raw = (A_flat * y_train_flat).reshape(1, -1)
    print(f"Shape of predictions_raw: {predictions_raw.shape}")
    predictions = None
    try:
        predictions = np.sign(np.dot(A_flat * y_train_flat, K_test))
        print(f"Shape of predictions: {predictions.shape}")
    except ValueError as e:
        print(f"Error with: {e}")
    print("Prediction completed.")
    
    if predictions is not None:
        return predictions.flatten()
    else: # If predictions couldn't be computed
        return np.zeros(y_train_flat.shape) 

def evaluate_accuracy(y_true, y_pred):
    print("Evaluating accuracy...")
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    print(f"Correct predictions: {correct_predictions}, Total samples: {len(y_true)}")
    print(f"Accuracy: {accuracy}")
    print("Evaluation completed.")
    return accuracy

def plot_with_test_set(X_train, y_train, X_test, y_pred_test, support_vectors):
    plt.figure(figsize=(10, 8))
    scatter(X_train[0, :], X_train[1, :], c=y_train, cmap=plt.cm.bwr, edgecolors='k', label='Training Data')
    scatter(X_train[0, support_vectors], X_train[1, support_vectors], c='yellow', edgecolors='k', label='Support Vectors', s=100)
    for i in range(len(X_test[0])):
        color = 'green' if y_pred_test[i] == y_test[i] else 'red'
        scatter(X_test[0, i], X_test[1, i], c=color, edgecolors='k', marker='x' if color == 'red' else 'o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Part 1: Simple simulation
polynomial_degrees = [2, 5, 10, 15]
gaussian_sigmas = [0.01, 0.25, 0.5, 0.75, 1.0]

X_train, y_train = checkboard(2000)
X_test, y_test = checkboard(1000)

A_poly = train_adatron(X_train.T, y_train, kernel_polynomial, d=3)
A_gauss = train_adatron(X_train.T, y_train, kernel_gaussian, sigma=0.5)

y_pred_poly = predict_adatron(X_train.T, X_test.T, A_poly, y_train, kernel_polynomial, d=3)
y_pred_gauss = predict_adatron(X_train.T, X_test.T, A_gauss, y_train, kernel_gaussian, sigma=0.5)

accuracy_poly = evaluate_accuracy(y_test, y_pred_poly)
accuracy_gauss = evaluate_accuracy(y_test, y_pred_gauss)

print(f"Polynomial Kernel Accuracy: {accuracy_poly}")
print(f"Gaussian Kernel Accuracy: {accuracy_gauss}")

test_checkboard()
last_epoch = -1

support_vectors_poly = A_poly[last_epoch].flatten() != 0
plot_with_test_set(X_train, y_train, X_test, y_pred_poly, support_vectors_poly)

support_vectors_gauss = A_gauss[last_epoch].flatten() != 0
plot_with_test_set(X_train, y_train, X_test, y_pred_gauss, support_vectors_gauss)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of support_vectors_poly: {support_vectors_poly.shape}")
print(f"Shape of support_vectors_gauss: {support_vectors_gauss.shape}")

try:
    plt.figure(figsize=(10, 8))
    scatter(X_train[0, support_vectors_poly], X_train[1, support_vectors_poly], c='g', edgecolors='k', label='Support Vectors Poly', s=100)
    scatter(X_train[0, support_vectors_gauss], X_train[1, support_vectors_gauss], c='y', edgecolors='k', label='Support Vectors Gauss', s=100)
    plt.title('Checkerboard Data with Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
except IndexError as e:
    print(f"Error in plotting support vectors: {e}")
show()

#%% Part 2: A more comprehensive simulation for non-linear dataset
def plot_with_test_set(kernel_type, degree_or_sigma, accuracy, X_train, y_train, X_test, y_pred_test, support_vectors):
    plt.figure(figsize=(10, 8))
    print("Shapes of data arrays:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_pred_test:", y_pred_test.shape)

    assert len(X_train) == len(y_train), "Mismatch in training data and labels size"
    assert len(X_test) == len(y_pred_test), "Mismatch in test data and predictions size"

    plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=30, label='Training Data')
   
    if support_vectors is not None:
        plt.scatter(X_train[support_vectors, 0], X_train[support_vectors, 1], 
                    s=150, facecolors='none', edgecolors='brown', label='Support Vectors')
    
    misclassified = y_test[:len(X_test)] != y_pred_test
    plt.scatter(X_test[misclassified, 0], X_test[misclassified, 1], 
                s=50, c='red', marker='x', label='Misclassified Test Points')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f"{kernel_type} Kernel (Degree {degree_or_sigma}) Accuracy: {accuracy}")
    plt.legend()
    plt.show()

def summarize_and_visualize(kernel_type, degree_or_sigma, accuracy, X_train, y_train, support_vectors, X_test, y_pred_test):
    print(f"{kernel_type} Kernel ({'Degree' if kernel_type == 'Polynomial' else 'Sigma'} {degree_or_sigma}) Accuracy: {accuracy}")
    plot_with_test_set(kernel_type, degree_or_sigma, accuracy, X_train, y_train, X_test, y_pred_test, support_vectors)

y_test = np.array(y_test)

for d in polynomial_degrees:
    A_poly = train_adatron(X_train.T, y_train, kernel_polynomial, d=d)
    y_pred_poly = predict_adatron(X_train.T, X_test.T, A_poly, y_train, kernel_polynomial, d=d)
    accuracy_poly = evaluate_accuracy(y_test, y_pred_poly)
    support_vectors_poly = A_poly[-1].flatten() != 0
    X_train_corrected = X_train.T if X_train.shape[0] < X_train.shape[1] else X_train
    X_test_corrected = X_test.T if X_test.shape[0] < X_test.shape[1] else X_test
    summarize_and_visualize('Polynomial', d, accuracy_poly, X_train_corrected, y_train, support_vectors_poly, X_test_corrected, y_pred_poly)

for sigma in gaussian_sigmas:
    A_gauss = train_adatron(X_train.T, y_train, kernel_gaussian, sigma=sigma)
    y_pred_gauss = predict_adatron(X_train.T, X_test.T, A_gauss, y_train, kernel_gaussian, sigma=sigma)
    accuracy_gauss = evaluate_accuracy(y_test, y_pred_gauss)
    support_vectors_gauss = A_gauss[-1].flatten() != 0
    X_train_corrected = X_train.T if X_train.shape[0] < X_train.shape[1] else X_train
    X_test_corrected = X_test.T if X_test.shape[0] < X_test.shape[1] else X_test
    summarize_and_visualize('Gaussian', sigma, accuracy_gauss, X_train_corrected, y_train, support_vectors_gauss, X_test_corrected, y_pred_gauss)


# %% Answer 2: XOR - Part 1
from scipy.optimize import minimize
import itertools

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def polynomial_kernel(x1, x2):
    return (np.dot(x1, x2) + 1)**2

def objective_function(alpha, K, y):
    return np.sum(alpha) - 0.5 * np.sum(alpha * alpha * y[:, None] * y[None, :] * K)

def constraint(alpha, y):
    return np.dot(alpha, y)

def decision_function(x, X, y, alpha, bias):
    return np.sign(sum(y[i] * alpha[i] * polynomial_kernel(X[:, i], x) for i in range(len(y))) + bias)

def sign_activation(x):
    return np.where(x >= 0, 1, -1)

X_xor = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]).T
y_xor = np.array([-1, 1, 1, -1])

P = X_xor.shape[1]
K = np.zeros((P, P))

for i in range(P):
    for j in range(P):
        K[i, j] = polynomial_kernel(X_xor[:, i], X_xor[:, j])
bounds = [(0, None) for _ in range(P)]
initial_alpha = np.zeros(P)
result = minimize(lambda a: -objective_function(a, K, y_xor), 
                  initial_alpha, 
                  bounds=bounds, 
                  constraints={'type': 'eq', 'fun': lambda a: constraint(a, y_xor)})

optimal_alpha = result.x if result.success else None
optimal_alpha
support_vector = X_xor[:, 0]
y_support = y_xor[0]
bias = y_support - sum(y_xor[i] * optimal_alpha[i] * polynomial_kernel(X_xor[:, i], support_vector) for i in range(P))
grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
grid = np.array([grid_x.ravel(), grid_y.ravel()])
decision_values = np.array([decision_function(x, X_xor, y_xor, optimal_alpha, bias) for x in grid.T])

plt.figure(figsize=(8, 6))
plt.contourf(grid_x, grid_y, decision_values.reshape(grid_x.shape), levels=[-1, 0, 1], alpha=0.3, colors=['red', 'blue'])
plt.scatter(X_xor[0, :], X_xor[1, :], c=y_xor, cmap='bwr', edgecolor='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary for XOR with SVM and Polynomial Kernel')
plt.show()

# %% Error Prone result for MLP
class SimpleMLP:
    def __init__(self):
        self.w_hidden = np.array([[1, 1], [-1, -1]])
        self.b_hidden = np.array([-0.5, 1.5])
        self.w_output = np.array([1, 1])
        self.b_output = np.array([-0.5])

    def forward_pass(self, x):
        z_hidden = np.dot(self.w_hidden, x) + self.b_hidden
        a_hidden = sign_activation(z_hidden)
        z_output = np.dot(self.w_output, a_hidden) + self.b_output
        a_output = sign_activation(z_output)
        return a_output

mlp = SimpleMLP()
X_xor = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]).T
mlp_outputs = np.array([mlp.forward_pass(x) for x in X_xor.T])
print("MLP Outputs for XOR inputs:", mlp_outputs)

grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
grid = np.array([grid_x.ravel(), grid_y.ravel()])
svm_decision_values = np.array([decision_function(x, X_xor, y_xor, optimal_alpha, bias) for x in grid.T])
mlp_decision_values = np.array([mlp.forward_pass(x) for x in grid.T])
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[0].contourf(grid_x, grid_y, svm_decision_values.reshape(grid_x.shape), levels=[-1, 0, 1], alpha=0.3, colors=['red', 'blue'])
ax[0].scatter(X_xor[0, :], X_xor[1, :], c=y_xor, cmap='bwr', edgecolor='k')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[0].set_title('SVM Decision Boundary for XOR')
ax[1].contourf(grid_x, grid_y, mlp_decision_values.reshape(grid_x.shape), levels=[-1, 0, 1], alpha=0.3, colors=['red', 'blue'])
ax[1].scatter(X_xor[0, :], X_xor[1, :], c=y_xor, cmap='bwr', edgecolor='k')
ax[1].set_xlabel('x1')
ax[1].set_ylabel('x2')
ax[1].set_title('MLP Decision Boundary for XOR')

plt.show()
# %% 2.3: XOR with Multilayer Perceptron - Part 2
class TwoLayerMLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.w_hidden1 = np.random.randn(input_size, hidden_size1)
        self.b_hidden1 = np.zeros(hidden_size1)
        self.w_hidden2 = np.random.randn(hidden_size1, hidden_size2)
        self.b_hidden2 = np.zeros(hidden_size2)
        self.w_output = np.random.randn(hidden_size2, output_size)
        self.b_output = np.zeros(output_size)

    def forward_pass(self, x):
        z_hidden1 = np.dot(x, self.w_hidden1) + self.b_hidden1
        a_hidden1 = sigmoid(z_hidden1)
        z_hidden2 = np.dot(a_hidden1, self.w_hidden2) + self.b_hidden2
        a_hidden2 = sigmoid(z_hidden2)
        z_output = np.dot(a_hidden2, self.w_output) + self.b_output
        a_output = sigmoid(z_output)
        return a_output, a_hidden2, a_hidden1

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                a_output, a_hidden2, a_hidden1 = self.forward_pass(X[i])
                error = y[i] - a_output
                d_output = error * sigmoid_derivative(a_output)
                error_hidden2 = d_output.dot(self.w_output.T)
                d_hidden2 = error_hidden2 * sigmoid_derivative(a_hidden2)
                error_hidden1 = d_hidden2.dot(self.w_hidden2.T)
                d_hidden1 = error_hidden1 * sigmoid_derivative(a_hidden1)
                self.w_output += self.learning_rate * np.outer(a_hidden2, d_output)
                self.b_output += self.learning_rate * d_output
                self.w_hidden2 += self.learning_rate * np.outer(a_hidden1, d_hidden2)
                self.b_hidden2 += self.learning_rate * d_hidden2
                self.w_hidden1 += self.learning_rate * np.outer(X[i], d_hidden1)
                self.b_hidden1 += self.learning_rate * d_hidden1

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model.forward_pass(z)[0] for z in Z])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap='RdBu', alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdBu', edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('MLP Decision Boundary for XOR')

input_size = 2
hidden_size1 = 4
hidden_size2 = 4
output_size = 1
mlp = TwoLayerMLP(input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.1)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

mlp.train(X_xor, y_xor, epochs=10000)
for i in range(X_xor.shape[0]):
    output, _, _ = mlp.forward_pass(X_xor[i])
    print(f"Input: {X_xor[i]}, Predicted: {output}, True: {y_xor[i]}")
    y_xor_plot = np.where(y_xor == 0, -1, 1)

plt.figure(figsize=(10, 8))
plot_decision_boundary(mlp, X_xor, y_xor_plot)
plt.show()

# %% 2.3: XOR with Multilayer Perceptron - Part 3
def relu_activation(z):
    return np.maximum(0, z)

def derivative_relu_activation(z):
    return (z > 0).astype(float)

def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid_activation(z):
    sz = sigmoid_activation(z)
    return sz * (1 - sz)

def he_initialization(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])

class TwoLayerMLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.w_hidden1 = he_initialization((input_size, hidden_size1))
        self.b_hidden1 = np.zeros(hidden_size1)
        self.w_hidden2 = he_initialization((hidden_size1, hidden_size2))
        self.b_hidden2 = np.zeros(hidden_size2)
        self.w_output = he_initialization((hidden_size2, output_size))
        self.b_output = np.zeros(output_size)

    def forward_pass(self, x):
        self.z_hidden1 = np.dot(x, self.w_hidden1) + self.b_hidden1
        self.a_hidden1 = relu_activation(self.z_hidden1)
        self.z_hidden2 = np.dot(self.a_hidden1, self.w_hidden2) + self.b_hidden2
        self.a_hidden2 = relu_activation(self.z_hidden2)
        self.z_output = np.dot(self.a_hidden2, self.w_output) + self.b_output
        self.a_output = sigmoid_activation(self.z_output)
        return self.a_output

    def backward_pass(self, x, y):
        output_error = y - self.a_output
        d_output = output_error * derivative_sigmoid_activation(self.z_output)
        error_hidden2 = np.dot(d_output, self.w_output.T)
        d_hidden2 = error_hidden2 * derivative_relu_activation(self.z_hidden2)
        error_hidden1 = np.dot(d_hidden2, self.w_hidden2.T)
        d_hidden1 = error_hidden1 * derivative_relu_activation(self.z_hidden1)

        self.w_output += self.learning_rate * np.outer(self.a_hidden2, d_output)
        self.b_output += self.learning_rate * d_output
        self.w_hidden2 += self.learning_rate * np.outer(self.a_hidden1, d_hidden2)
        self.b_hidden2 += self.learning_rate * d_hidden2
        self.w_hidden1 += self.learning_rate * np.outer(x, d_hidden1)
        self.b_hidden1 += self.learning_rate * d_hidden1

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                self.forward_pass(X[i])
                self.backward_pass(X[i], y[i])

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.array([model.forward_pass(np.array([x, y])) for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)
    plt.show()
    
def compute_boundries():
    input_size = 2
    hidden_size1 = 4
    hidden_size2 = 4
    output_size = 1
    mlp_sign = TwoLayerMLP(input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.1)
    epochs = 10000

    mlp_sign.train(X_xor, y_xor, epochs)
    plot_decision_boundary(mlp_sign, X_xor, y_xor)
    predictions = np.round(mlp_sign.forward_pass(X_xor))
    print("Predictions after training:")
    print(predictions.flatten())
    accuracy = np.mean(predictions.flatten() == y_xor.flatten())
    print("Accuracy:", accuracy)

compute_boundries()
# %% Run again for 100% accuracy
compute_boundries()