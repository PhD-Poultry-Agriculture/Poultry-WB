# Author: German Shiklov
# ID: 317634517

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
#%% 3.1 Binary Perceptron

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
        # Visualize weight changes over epochs
        # fig, ax = plt.subplots()
        # ax.plot(w)
        # ax.set_title(f'Epoch {epochs}')
        # plt.show()

    return [w, did_converge, epochs]

N = 50  # Number of features
P = 100  # Number of samples

num_runs = 100
num_features = [10, 20, 100, 500, 1000]

for n in num_features:
    converged_count = 0
    epochs_count = 0
    
    for i in range(0,num_runs):
        X = np.random.choice([-1, 1], (n, P))
        y0 = np.random.choice([-1, 1], P)
        w, converged, epochs = perceptron(X, y0)
        
        # print("Weights:", w, "Converged:", converged, "Epochs:", epochs)
        epochs_count += epochs
        converged_count += 1 if converged else 0
    print("N:", n, "Converged:", converged_count/num_runs, "Epochs Avg:", epochs_count/num_runs)


# %% Part 3.2
num_runs = 100
num_features = [10, 20, 100, 500, 1000]
P = 100

results = {}

for n in num_features:
    epochs_list = []
    converged_count = 0
    
    for _ in range(num_runs):
        X = np.random.choice([-1, 1], (n, P))
        y0 = np.random.choice([-1, 1], P)
        w, converged, epochs = perceptron(X, y0)
        
        epochs_list.append(epochs)
        converged_count += 1 if converged else 0

    avg_epochs = np.mean(epochs_list)
    std_epochs = np.std(epochs_list)
    min_epochs = np.min(epochs_list)
    max_epochs = np.max(epochs_list)
    convergence_rate = converged_count / num_runs

    results[n] = {
        'Convergence Rate': convergence_rate,
        'Average Epochs': avg_epochs,
        'Std Dev of Epochs': std_epochs,
        'Min Epochs': min_epochs,
        'Max Epochs': max_epochs
    }

for n, stats in results.items():
    print(f'N = {n}:')
    for key, value in stats.items():
        print(f'  {key}: {value}')
    print()

#%% Part 3.3

def perceptron(X, y0): # Updated for better analysis
    t_max = 100
    N, P = X.shape
    w = np.zeros(N)
    w_history = []
    did_converge = False

    for epochs in range(1, t_max + 1):
        w_old = np.copy(w)
        w_history.append(w_old)

        for i in range(P):
            if y0[i] * np.dot(w, X[:, i]) <= 0:
                w = w + y0[i] * X[:, i]

        if np.array_equal(w, w_old):
            did_converge = True
            break

    w_history = np.array(w_history)
    return [w, did_converge, epochs, w_history]

def analyze_perceptron_results(num_features, num_runs, P):
    results = {}
    for n in num_features:
        epochs_list = []
        converged_count = 0
        weight_covariances = []
        weight_correlations = []
        
        for _ in range(num_runs):
            X = np.random.choice([-1, 1], (n, P))
            y0 = np.random.choice([-1, 1], P)
            w, converged, epochs, w_history = perceptron(X, y0)
            
            epochs_list.append(epochs)
            converged_count += 1 if converged else 0
            
            if converged:
                cov_w = np.cov(w_history, rowvar=False)
                corr_w = np.corrcoef(w_history, rowvar=False)
                
                weight_covariances.append(cov_w)
                weight_correlations.append(corr_w)

        avg_epochs = np.mean(epochs_list)
        std_epochs = np.std(epochs_list)
        median_epochs = np.median(epochs_list)
        iqr_epochs = np.percentile(epochs_list, 75) - np.percentile(epochs_list, 25)
        convergence_rate = converged_count / num_runs

        results[n] = {
            'Convergence Rate': convergence_rate,
            'Average Epochs': avg_epochs,
            'Std Dev of Epochs': std_epochs,
            'Median Epochs': median_epochs,
            'IQR of Epochs': iqr_epochs,
            'Average Weight Covariance': np.mean(weight_covariances) if weight_covariances else None,
            'Average Weight Correlation': np.mean(weight_correlations) if weight_correlations else None
        }

    return results

num_features = [10, 20, 100, 500, 1000]
P = 10
num_runs = 100

results = analyze_perceptron_results(num_features, num_runs, P)
for n, stats in results.items():
    print(f'N = {n}:')
    for key, value in stats.items():
        print(f'  {key}: {value}')
    print()

# %%
def run_simulation(N, alpha_range, num_runs, t_max):
    convergence_rates = np.zeros(len(alpha_range))
    for i, alpha in enumerate(alpha_range):
        P = int(round(alpha * N))
        converged_count = 0
        for _ in range(num_runs):
            X = np.random.choice([-1, 1], (N, P))
            y0 = np.random.choice([-1, 1], P)
            _, converged, epochs, _ = perceptron(X, y0)
            if converged and epochs <= t_max:
                converged_count += 1
        convergence_rates[i] = converged_count / num_runs
    return convergence_rates

N_values = [10, 20, 50, 100]
alpha_range = np.arange(0, 3.1, 0.1)
num_runs = 100
t_max_values = [1000, 10000, 100000]

for t_max in t_max_values:
    plt.figure(figsize=(12, 8))
    for N in N_values:
        convergence_rates = run_simulation(N, alpha_range, num_runs, t_max)
        plt.plot(alpha_range, convergence_rates, label=f'N = {N}')
    plt.title(f'Probability of Convergence for t_max = {t_max}')
    plt.xlabel('Alpha (P/N)')
    plt.ylabel('Convergence Probability')
    plt.legend()
    plt.show()
# %%
