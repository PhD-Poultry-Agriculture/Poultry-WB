#%%
# Author: German Shiklov
# ID: 317634517
# Neural Link - Exercise 5.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import zeros
from numpy.random import randint
from pylab import *
import numpy as np

__all__ = [ 'plot_board' ]
UP, RIGHT, DOWN, LEFT = range(4)
num_episodes = 1000
grid_shape = (4, 12)
start_state = (3, 0)
goal_state = (3, 11)
cliff_states = [(3, i) for i in range(1, 11)]
epsilon_range = np.linspace(0.1, 1.0, 10)
epsilon_values = np.linspace(0.1, 0.9, 5)
learning_rate = 0.1
gamma = 1.0

def plot_board(h,w,s,pi,Q):
    # parameters:
    # h,w: height+width of the board
    # s: current state (red dot)
    # pi: policy
    # Q: Q values
    print(f"Current state (red dot) coordinates: {s}")
    print(f"Coordinates for arrows (policy) and Q-values are expected to start from the top-left corner")

    ax = subplot(211)
    ax.clear()
    ax.hlines(range(0,h+1),0,w,color='k')
    ax.vlines(range(0,w+1),0,h,color='k')
    ax.set_ylim([-0.1,h+.1])
    ax.set_xlim([-.1,w+.1])
    ax.add_patch(Rectangle((1,0),w-2,1,fc='k',alpha=.5))
    ax.set_axis_off()
    
    aw=0.3
    for sx in range(w):
        for sy in range(h):
            if sy!=0 or (sx==0):
                a = pi[sy,sx]
                if a == 0:#up
                    ax.arrow(sx+0.5,sy+0.5-aw,0,aw,width=0.05,head_starts_at_zero=True,alpha=0.5)
                elif a == 2:#down
                    ax.arrow(sx+0.5,sy+0.5+aw,0,-aw,width=0.05,head_starts_at_zero=True,alpha=0.5)
                elif a == 1:#right
                    ax.arrow(sx+0.5-aw,sy+0.5,aw,0,width=0.05,head_starts_at_zero=True,alpha=0.5)
                elif a == 3:#left
                    ax.arrow(sx+0.5+aw,sy+0.5,-aw,0,width=0.05,head_starts_at_zero=True,alpha=0.5)                
   
    ax.set_aspect('equal')
    ax1=subplot(212)
    ax1.clear()
    ax1.hlines(range(0,h+1),0,w,color='k')
    ax1.vlines(range(0,w+1),0,h,color='k')
    ax1.set_ylim([-0.1,h+.1])
    ax1.set_xlim([-.1,w+.1])
    ax1.add_patch(Rectangle((1,0),w-2,1,fc='k',alpha=.5))
    ax1.set_axis_off()
    ax1.set_aspect('equal')
    for sx in range(w):
        for sy in range(h):
            if sy!=0 or (sx==0):
                ax1.annotate('{:.3}'.format(max(Q[sy,sx,:])),(sx+0.5,sy+0.5),ha='center')
    
    ax.text(goal_state[1] + 0.5, h - goal_state[0] - 0.5, 'G', color='gold', ha='center', va='center', fontsize=12)
    ax.scatter(s[1]+0.5, h-s[0]-0.5, s=125, color='r')

    return ax,ax1

def take_action(state, action, goal_state, cliff_states, grid_shape):
    """Returns the next state and reward given an action."""
    next_state = list(state)
    if action == UP and state[0] > 0:
        next_state[0] -= 1
    elif action == RIGHT and state[1] < grid_shape[1] - 1:
        next_state[1] += 1
    elif action == DOWN and state[0] < grid_shape[0] - 1:
        next_state[0] += 1
    elif action == LEFT and state[1] > 0:
        next_state[1] -= 1
    next_state = tuple(next_state)
    
    if next_state in cliff_states:
        print(f"Fell off the cliff at: {next_state}")
        return start_state, -100
    elif next_state == goal_state:
        print(f"Reached goal at: {next_state}")
        return next_state, 0
    else:
        return next_state, -1
    
def epsilon_greedy_policy(state, Q, epsilon):
    """Epsilon-greedy policy for action selection based on a given epsilon."""
    num_actions = len(Q[state])
    greedy_action = np.argmax(Q[state])
    probabilities = np.full(num_actions, (1 - epsilon) / num_actions)
    probabilities[greedy_action] += epsilon
    return np.random.choice(np.arange(num_actions), p=probabilities)

def simulate_q_learning_epsilon_dependent(Q, start_state, goal_state, cliff_states, grid_shape, epsilon_range, episodes, lr, gamma):
    mean_episode_lengths = []
    for epsilon in epsilon_range:
        Q = np.zeros(grid_shape + (4,))
        episode_lengths = []
        for _ in range(episodes):
            state = start_state
            episode_length = 0
            while state != goal_state:
                action = epsilon_greedy_policy(state, Q, epsilon)
                next_state, reward = take_action(state, action, goal_state, cliff_states, grid_shape)
                Q[state][action] += lr * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
                state = next_state
                episode_length += 1
                if next_state == goal_state:
                    print(f"Reached goal at: {next_state}")
                    if Q[goal_state].sum() == 0:
                        Q[goal_state][:] = 0
                    break

            episode_lengths.append(episode_length)
        mean_episode_lengths.append(np.mean(episode_lengths))

    return mean_episode_lengths

mean_episode_lengths = simulate_q_learning_epsilon_dependent(
    Q=np.zeros(grid_shape + (4,)),
    start_state=start_state,
    goal_state=goal_state,
    cliff_states=cliff_states,
    grid_shape=grid_shape,
    epsilon_range=epsilon_range,
    episodes=num_episodes,
    lr=learning_rate,
    gamma=gamma)

plt.figure(figsize=(10, 6))
plt.plot(epsilon_range, mean_episode_lengths, '-o', label='Mean Episode Length')
plt.xlabel('Epsilon')
plt.ylabel('Mean Episode Length')
plt.xscale('linear')
plt.yscale('linear')
plt.title('Mean Episode Length as a Function of Epsilon')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# %%
def simulation_2(epsilon_values, num_episodes, learning_rate, gamma, grid_shape, start_state, goal_state, cliff_states):
    average_rewards = {eps: [] for eps in epsilon_values}
    Q = None
    for epsilon in epsilon_values:
        Q = np.zeros(grid_shape + (4,))
        total_reward = 0
        for episode in range(1, num_episodes + 1):
            state = start_state
            while state != goal_state:
                action = epsilon_greedy_policy(state, Q, epsilon)
                next_state, reward = take_action(state, action, goal_state, cliff_states, grid_shape)
                Q[state][action] += learning_rate * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
                state = next_state
                total_reward += reward
                if next_state == goal_state:
                    print(f"Reached goal at: {next_state}")
                    if np.sum(Q[next_state]) == 0:
                        reward_for_goal = 0
                        Q[next_state] = reward_for_goal
                        print(f"Update Q at goal: {Q[next_state]}")
                    break 
            average_rewards[epsilon].append(total_reward / episode)
    
    plt.figure(figsize=(12, 8))
    for epsilon in epsilon_values:
        plt.plot(average_rewards[epsilon], label=f'epsilon = {epsilon:.2f}')
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (E[Rn])')
    plt.title('Average Reward as a Function of Time (Episodes)')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    return Q

Q_values = simulation_2(epsilon_values, num_episodes, learning_rate, gamma, grid_shape, start_state, goal_state, cliff_states)
# %% Bonus
policy = np.argmax(Q_values, axis=2)
policy[goal_state] = -1

def simulation_3(grid_shape, start_state, goal_state, cliff_states, Q_values):
    policy = np.argmax(Q_values, axis=2)    
    print("Policy shape:", policy.shape)
    print("Policy at goal state:", policy[goal_state])
    print("Goal state:", goal_state)
    print("Q-values at goal state:", Q_values[goal_state])
    
    plot_board(grid_shape[0], grid_shape[1], goal_state, policy, Q_values)

simulation_3(grid_shape, start_state, goal_state, cliff_states, Q_values)
