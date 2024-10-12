# Author: German Shiklov
# ID: 317634517
# Neural Link - Exercise 5.
#%%
import numpy as np
import matplotlib.pyplot as plt

def random(isGaussian):
    return np.random.rand() if isGaussian else np.random.normal()

def simulate_bandit(PL, PR, eta, num_iterations=1000, initial_p=0.5):
    p_record = np.zeros(num_iterations)
    p = initial_p
    epsilon = 1e-8
    for t in range(num_iterations):
        p_record[t] = p
        # Distribution: isGaussian = False, then Uniform
        isGaussian = True
        
        choose_right = random(isGaussian) < p
        
        if choose_right:
            reward = random(isGaussian) < PR
            # p_gradient = ((1 if reward else 0) - PR) / (p+epsilon)
            # p_gradient = (reward - PR) / (p+epsilon)
            # p_gradient = (1 - PR) if reward else -PR     
            # p_gradient = ((1 if reward else 0) - PR) / (p+epsilon)
            p_gradient = (1 / (p+epsilon)) if reward else (-1 / (p+epsilon))
        else:
            reward = random(isGaussian) < PL
            # p_gradient = (PL - (1 if reward else 0)) / (1 - p+epsilon)
            # p_gradient = (PL - reward) / (1 - p+epsilon)
            # p_gradient = -PL if reward else (1 - PL)
            # p_gradient = (PL - (1 if reward else 0)) / (1 - p+epsilon)
            p_gradient = (-1 / (1 - p+epsilon)) if reward else (1 / (1 - p+epsilon))

        # p += eta * p_gradient
        p += eta * p_gradient * (reward - (PL * (1 - p) + PR * p))
        p = max(min(p, 1), 0)
    
    return p_record

scenarios = {
    'Simulation-4': {'PL': 0.2, 'PR': 0.7, 'eta': 0.01},
    'Simulation-5': {'PL': 0.4, 'PR': 0.5, 'eta': 0.01},
    'Simulation-6a': {'PL': 0.2, 'PR': 0.7, 'eta': 0.025},
    'Simulation-6b': {'PL': 0.4, 'PR': 0.5, 'eta': 0.025}
}

plt.figure(figsize=(14, 7))
for scenario, params in scenarios.items():
    p_record = simulate_bandit(**params)
    label = f'{scenario}: $P_L={params["PL"]}$, $P_R={params["PR"]}$, LR={params["eta"]}'
    plt.plot(p_record, label=label)

plt.xlabel('Iterations')
plt.ylabel('Probability p of choosing right machine')
plt.title('Evolution of probability p over iterations')
plt.legend()
plt.show()


# %%
