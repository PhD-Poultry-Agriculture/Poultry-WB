# Author: German Shiklov
# ID: 317634517

#%% Imports
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#%% 2 Bayesian Decision Making
prior_probabilities = {'Good': 0.2, 'Mediocre': 0.4, 'Bad': 0.4}
lecture_given_course = {
    'Interesting': {'Good': 0.8, 'Mediocre': 0.5, 'Bad': 0.1},
    'Boring': {'Good': 0.2, 'Mediocre': 0.5, 'Bad': 0.9}
}
risk = {
    'Attending': {'Good': 0, 'Mediocre': 7, 'Bad': 10},
    'Not Attending': {'Good': 20, 'Mediocre': 5, 'Bad': 0}
}

# Total probability of each lecture type
total_probability_lecture = {lecture_type: sum(lecture_given_course[lecture_type][q] * prior_probabilities[q] for q in prior_probabilities) for lecture_type in lecture_given_course}

# Conditional probabilities of course quality given lecture quality
conditional_probabilities = {}
for lecture_type in lecture_given_course:
    conditional_probabilities[lecture_type] = {}
    for quality in prior_probabilities:
        conditional_probabilities[lecture_type][quality] = (lecture_given_course[lecture_type][quality] * prior_probabilities[quality]) / total_probability_lecture[lecture_type]

# Conditional risk for each action given lecture quality
conditional_risk = {}
for action in risk:
    conditional_risk[action] = {}
    for lecture_type in lecture_given_course:
        conditional_risk[action][lecture_type] = sum(risk[action][q] * conditional_probabilities[lecture_type][q] for q in prior_probabilities)

# Determine optimal strategy and its total risk
optimal_strategy = {}
total_risk = 0
for lecture_type in lecture_given_course:
    best_action = min(conditional_risk, key=lambda a: conditional_risk[a][lecture_type])
    optimal_strategy[lecture_type] = best_action
    total_risk += total_probability_lecture[lecture_type] * conditional_risk[best_action][lecture_type]

conditional_risk, optimal_strategy, total_risk

# %% 4) Non-bayesian Decision-making - Part 1
loss = np.array([
    [0, 7, 10],     # Losses when attending [Good, Mediocre, Bad]
    [20, 5, 0]      # Losses when not attending [Good, Mediocre, Bad]
])

# p(x|ω) for each state of the world
# x = [Interesting, Boring], ω = [Good, Mediocre, Bad]
conditional_prob = np.array([ # Interesting or Boring given some State
    [0.8, 0.2],
    [0.5, 0.5],
    [0.1, 0.9]
])

# conditional risk R[δ|ω] for each action and state
state_conditional_risk = np.dot(loss, conditional_prob)

# minimax decision rule: min over actions, max over states
minimax_decision = np.argmin(np.max(state_conditional_risk, axis=1))

decision_mapping = ['Attending', 'Not Attending']
minimax_decision_str = decision_mapping[minimax_decision]
minimax_decision_str, state_conditional_risk


# %% 4) Non-bayesian Decision-making - Part 2
# Stochastic policy π
# π(Attending|Interesting) = 0.9, π(Not Attending|Interesting) = 0.1
# π(Attending|Boring) = 0, π(Not Attending|Boring) = 1
pi = np.array([
    [0.9, 0.1],
    [0.0, 1.0]
])

# R[π|ω]
state_conditional_risk_pi = np.dot(loss, np.dot(conditional_prob, pi.T))
state_conditional_risk_pi

# %% (5) Decision Boundries between two normal distributions
sigma_0 = 1
sigma_1 = 2

x_values = np.linspace(-10, 10, 400)

# P(x|ω0) and P(x|ω1)
px_given_omega0 = norm.pdf(x_values, 0, sigma_0)
px_given_omega1 = norm.pdf(x_values, 0, sigma_1)

# P(ω0) and P(ω1)
p_omega0 = p_omega1 = 0.5

# P(x)
px = px_given_omega0 * p_omega0 + px_given_omega1 * p_omega1

# Posterior probabilities P(ω0|x) and P(ω1|x)
p_omega0_given_x = (px_given_omega0 * p_omega0) / px
p_omega1_given_x = (px_given_omega1 * p_omega1) / px

# Expected cost for each action α0 and α1
# Cost function λ(αi|ωj) = 1 - δij, so the cost is 0 for correct decision, 1 for incorrect
expected_cost_a0 = p_omega1_given_x
expected_cost_a1 = p_omega0_given_x

optimal_action = np.where(expected_cost_a0 < expected_cost_a1, 0, 1)  # Choose α0 if its expected cost is lower, else α1
optimal_action, expected_cost_a0, expected_cost_a1

x_values = np.linspace(-10, 10, 400)

# P(x|ω0) and P(x|ω1)
px_given_omega0 = norm.pdf(x_values, 0, sigma_0)
px_given_omega1 = norm.pdf(x_values, 0, sigma_1)

px = px_given_omega0 * p_omega0 + px_given_omega1 * p_omega1

p_omega0_given_x = (px_given_omega0 * p_omega0) / px
p_omega1_given_x = (px_given_omega1 * p_omega1) / px

expected_cost_a0 = p_omega1_given_x
expected_cost_a1 = p_omega0_given_x

optimal_action = np.where(expected_cost_a0 < expected_cost_a1, 0, 1)  # Choose α0 if its expected cost is lower, else α1
optimal_action

# %% Answer (6)
def read_distribution(filename):
    distribution = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 2:
                letter = line[0].lower()
                freq = float(line[1])
                distribution[letter] = freq
    return distribution

def read_text(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read().lower()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            return f.read().lower()

def sample_and_update_posteriors(text, N, english_dist, french_dist, priors):
    posteriors = priors.copy()
    posterior_history = {'English': [], 'French': []}
    small_prob = 1e-6  # Some epsilon to prevent division by zero

    for i in range(N):
        sampled_letter = random.choice(text)

        likelihood_english = english_dist.get(sampled_letter, small_prob)
        likelihood_french = french_dist.get(sampled_letter, small_prob)

        posteriors['English'] *= likelihood_english
        posteriors['French'] *= likelihood_french

        # Normalize posteriors
        total = sum(posteriors.values()) + 1e-6
        posteriors = {k: v / total for k, v in posteriors.items()}

        posterior_history['English'].append(posteriors['English'])
        posterior_history['French'].append(posteriors['French'])

        if i % 10 == 0:
            print(f"Sample #{i}: Letter = {sampled_letter}, Posteriors = {posteriors}")

    return posterior_history
#%%

def plot_posteriors(posterior_history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(posterior_history['English'], label='P(English|x1,...,xn)')
    plt.plot(posterior_history['French'], label='P(French|x1,...,xn)')
    plt.xlabel('Number of Samples')
    plt.ylabel('Posterior Probability')
    plt.title(title)
    plt.legend()
    plt.show()

#%%
import os

# Get the current working directory
current_working_directory = os.getcwd()

# Get the directory of the script file
script_directory = os.path.dirname(os.path.abspath(__file__))

# Print both paths
print("Current working directory:", current_working_directory)
print("Directory of the script file:", script_directory)
#%%

english_dist = read_distribution('InfoEncodeInBrain/english_dist.txt')
french_dist = read_distribution('InfoEncodeInBrain/french_dist.txt')
english_text = read_text('InfoEncodeInBrain/english_text.txt')
french_text = read_text('InfoEncodeInBrain/french_text.txt')

print("English Distribution Snippet:", list(english_dist.items())[:])
print("French Distribution Snippet:", list(french_dist.items())[:])
print("English Text Snippet:", english_text[:100])
print("French Text Snippet:", french_text[:100])

priors = {'English': 0.7, 'French': 0.3}
N = 100

print("\nSampling from English text...")
posterior_history_english = sample_and_update_posteriors(english_text, N, english_dist, french_dist, priors)
print("\nSampling from French text...")
posterior_history_french = sample_and_update_posteriors(french_text, N, english_dist, french_dist, priors)
plot_posteriors(posterior_history_english, 'Posterior Probabilities in English Text')
plot_posteriors(posterior_history_french, 'Posterior Probabilities in French Text')

# %%
import os

# Get the current working directory
current_path = os.getcwd()

# Print the current path
print("Current path:", current_path)

# %%
