import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Convolutional layer
x_image = tf.reshape(x, [-1,28,28,1])
W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 30, 500], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
h_pool_flat = tf.reshape(h_pool, [-1, 12 * 12 * 30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_hat = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = - tf.reduce_sum(y_*tf.log(y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
print("=While Training==================")
print("=================================")
print("|Epoch\tBatch\t|")
print("|===============================|")
for j in range(5):
    for i in range(550):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        if i % 110 == 109:
            print("|%d\t|%d\t" % (j + 1, i + 1))
print("|===============================|")

# For gathering hidden layer output
Psi = sess.run(h_fc1, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
Phi = Psi - np.mean(Psi, axis=0)
U, s, V = np.linalg.svd(np.matmul(np.transpose(Phi), Phi), full_matrices=True)

Z = np.matmul(Phi, U)
x1 = Z[:, 0]
x2 = Z[:, 1]

# Construct a new dataset with 100 examples
new_images_dataset = []
new_labels_dataset = []

c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = c8 = c9 = 0
for i, lab in enumerate(mnist.test.labels[:1000]):
    if c0 < 10:
        if np.argmax(lab) in [0]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c0 = c0 + 1
    else:
        break
for i, lab in enumerate(mnist.test.labels[1000:2000]):
    if c1 < 10:
        if np.argmax(lab) in [1]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c1 = c1 + 1
    else:
        break
for i, lab in enumerate(mnist.test.labels[2000:3000]):
    if c2 < 10:
        if np.argmax(lab) in [2]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c2 = c2 + 1
    else:
        break
for i, lab in enumerate(mnist.test.labels[3000:4000]):
    if c3 < 10:
        if np.argmax(lab) in [3]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c3 = c3 + 1
    else:
        break
for i, lab in enumerate(mnist.test.labels[4000:5000]):
    if c4 < 10:
        if np.argmax(lab) in [4]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c4 = c4 + 1
    else:
        break
for i, lab in enumerate(mnist.test.labels[5000:6000]):
    if c5 < 10:
        if np.argmax(lab) in [5]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c5 = c5 + 1
    else:
        break
for i, lab in enumerate(mnist.test.labels[6000:7000]):
    if c6 < 10:
        if np.argmax(lab) in [6]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c6 = c6 + 1
    else:
        break
for i, lab in enumerate(mnist.test.labels[7000:8000]):
    if c7 < 10:
        if np.argmax(lab) in [7]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c7 = c7 + 1
    else:
        break
for i, lab in enumerate(mnist.test.labels[8000:9000]):
    if c8 < 10:
        if np.argmax(lab) in [8]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c8 = c8 + 1
    else:
        break
for i, lab in enumerate(mnist.test.labels[9000:]):
    if c9 < 10:
        if np.argmax(lab) in [9]:
            new_images_dataset.append(mnist.test.images[i])
            new_labels_dataset.append(mnist.test.labels[i])
            c9 = c9 + 1
    else:
        break

# Get the output of the hidden layer from new dataset
Omega = sess.run(h_fc1, feed_dict={x: new_images_dataset, y_: new_labels_dataset})
print(Omega.shape)

Q = np.matmul(Omega, U)

x1_new = Q[:, 0]
x2_new = Q[:, 1]

# Plot the first two columns of Z
plt.scatter(x1, x2, facecolor='green', edgecolor="green", s=20, label="First two columns from Z = Phi W")
plt.scatter(x1_new, x2_new, facecolor='blue', edgecolor="blue", s=20, label="First two columns from Q = Omega W")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.title("EE488 Project#1 task#3 20080242 Hyuk Jun Kim")
plt.legend(loc="best")
plt.savefig('plt_task3.png')