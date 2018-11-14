# 20080242 Kim Hyuk Jun
# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Project #2 Task #1: Linear Environment
# Date: Nov 29. 2017

import numpy as np
import matplotlib.pyplot as plt
from qlearning import *

# environment with 20 states where 2 of them are terminal states
class linear_environment:
    def __init__(self):
        self.n_states = 21       # number of states
        self.n_actions = 2       # number of actions

        next_state = [[i-1, i+1] for i in range(self.n_states)]
        next_state[0], next_state[-1] = [0,0], [20,20]
        self.next_state = np.array(next_state, dtype=np.int)

        self.reward = np.zeros((self.n_states, 2))
        self.reward[1] = [1., 0.]
        self.reward[-2] = [0., 1.]
        self.terminal = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], dtype=np.int)
        self.init_state = 10     # initial state

# an instance of the environment
env = linear_environment()

# For the first sub-problem, with epsilon = 1
# n_episodes = 1     # Set for sub-problem 1-1
# n_episodes = 5     # Set for sub-problem 1-2
n_episodes = 1000    # Set for sub-problem 1-3
# n_episodes = 100     # Set for sub-problem 2-1
max_steps = 1000     # max. # of steps to run in each episode
alpha = 0.2          # learning rate
gamma = 0.9          # discount factor

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 1.   # Set for epsilon = 1
# epsilon.final = 0.   # Set for sub-problem 2-1, with epsilon = 1 initially and linearly reduce to 0
epsilon.dec_episode = 1. / n_episodes  # amount of decrement in each episode
epsilon.dec_step = 0.                  # amount of decrement in each step

Q, n_steps, sum_rewards = Q_learning_train(env, n_episodes, max_steps, alpha, gamma, epsilon)
print('Q(s,a)')
print(Q)
# for k in range(n_episodes):
    # print('%2d: %.2f' % (k, sum_rewards[k]))

# Print for obtain average numbers of stpes in sub-problem 1-3
print('average_numbers_of_steps = %.2f' % np.mean(n_steps))

# for plotting in sub-problem 2
plt.plot(range(n_episodes), n_steps, label='linear.py')
plt.legend(loc="best")
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
# plt.savefig('linear.py.png')

test_n_episodes = 1     # number of episodes to run
test_max_steps = 1000   # max. # of steps to run in each episode
test_epsilon = 0.       # test epsilon
test_n_steps, test_sum_rewards, s, a, sn, r = Q_test(Q, env, test_n_episodes, test_max_steps, test_epsilon)
# Print for obtain test_n_steps[0] in sub-problem 1-1, 1-2
print('test_n_steps[0] = %.2f' % test_n_steps[0])
print('test_n_rewards[0] = %.2f' % test_sum_rewards[0])
