# 20080242 Kim Hyuk Jun
# EE488B Special Topics in EE <Deep Learning and AlphaGo>, Fall 2017
# Project #2 Task #2: 4-legged spider
# Date: Nov 29. 2017


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from breakout_env import *
from breakout_ani import *
import random
import tensorflow as tf
from wait import *

# Breakout Environment
env = breakout_environment(5, 8, 3, 1, 2)

n_episodes = 100     # number of episodes to run, 1 for continuing task
max_steps = 200      # max. # of steps to run in each episode
alpha = 0.2          # learning rate
gamma = 0.9          # discount factor

batch_size = 32
max_memory = 1000

class epsilon_profile: pass
epsilon = epsilon_profile()
epsilon.init = 1.    # initial epsilon in e-greedy
epsilon.final = 0.   # final epsilon in e-greedy
epsilon.dec_episode = 0.  # amount of decrement in each episode
epsilon.dec_step = 1. / max_steps   # amount of decrement in each step

x = tf.placeholder(tf.float32, shape=[None, env.ny, env.nx, env.nf])
y_ = tf.placeholder(tf.float32, shape=[None, env.na])

W1 = tf.Variable(tf.truncated_normal([env.ny, env.nx, env.nf, 30], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([30], stddev=0.01))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([30, 30], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([30], stddev=0.01))
h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([30, env.na], stddev=0.1))
b3 = tf.Variable(tf.truncated_normal([env.na], stddev=0.01))
y_hat = tf.nn.relu(tf.matmul(h2, W3) + b3)

# Train and Evaluate the Model
cost = tf.reduce_mean(tf.square(y_ - y_hat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost)

class ReplayMemory:
    def __init__(self, nx, ny, max_memory, discount):
        self.max_memory = max_memory
        self.nx = nx
        self.ny = ny
        self.discount = discount
        canvas = np.zeros((self.nx, self.ny))
        canvas = np.reshape(canvas, (-1, self.nx * self.ny))
        self.input_state = np.empty((self.max_memory, 100), dtype=np.float32)
        self.actions = np.zeros(self.max_memory, dtype=np.uint8)
        self.next_state = np.empty((self.max_memory, 100), dtype=np.float32)
        self.rewards = np.empty(self.max_memory, dtype=np.int8)
        self.count = 0
        self.current = 0

    def remember(self, current_state, action, reward, next_state):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.input_state[self.current] = current_state
        self.next_state[self.current] = next_state
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.max_memory

    def getBatch(self, model, batch_size, na, nx, ny, sess, x):
        memory_length = self.count
        chosen_batch_size = min(batch_size, memory_length)

        inputs = np.zeros((chosen_batch_size, nx * ny))
        targets = np.zeros((chosen_batch_size, na))

        for i in xrange(chosen_batch_size):
            if memory_length == 1:
                memory_length = 2

            rindex = random.randrange(1, memory_length)
            current_input_state = np.reshape(self.input_state[rindex], (1, 100))

            target = sess.run(model, feed_dict={x: current_input_state})

            current_next_state = np.reshape(self.next_state[rindex], (1, 100))
            current_outputs = sess.run(model, feed_dict={x: current_next_state})

            next_state_max_Q = np.amax(current_outputs)
            target[0, [self.actions[rindex] - 1]] = self.rewards[rindex] + self.discount * next_state_max_Q

            inputs[i] = current_input_state
            targets[i] = target

        return inputs, targets

memory = ReplayMemory(env.nx, env.ny, max_memory, gamma)

win_cnt = 0

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initialize())
saver = tf.train.Saver()

for episode in range(n_episodes):
    err = 0
    s = env.reset()
    for t in range(max_steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(env.na)   # random action
        else:
            q = sess.run(y_hat, feed_dict={x: np.reshape(s, [1, env.ny, env.nx, env.nf])})
            a = np.random.choice(np.where(q[0] == np.max(q))[0])
        sn, r, terminal, _, _, _, _, _, _, _, _ = env.run(a - 1)  # action to take is -1, 0, 1

        if (r == 1):
            win_cnt = win_cnt + 1

        memory.remember(s, a, r, sn)

        s = sn
        inputs, targets = memory.getBatch(y_hat, batch_size, env.na, env.nx, env.ny, sess, x)

        _, loss = sess.run([optimizer, cost], feed_dict={x: inputs, y_: targets})
        err = err + loss
    print("Episode " + str(i) + ": err = " + str(err) + ": Win count = " + str(win_cnt) + " Win ratio = " + str(float(win_cnt) / float(t + 1) * 100))

save_path = saver.save(sess, "./breakout.ckpt")

ani = breakout_animation(env, 20)
ani.save('breakout.mp4', dpi=200)
# plt.show(block=False)
wait('Press enter to quit')

