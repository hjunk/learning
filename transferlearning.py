import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
Y_before = tf.placeholder(tf.float32, shape=[None, 9])    # : Change number of output categories from 10 to 9

# Conditions & generate filtered dataset
selected_labels = [0, 1, 2, 4, 5, 6, 7, 8, 9]   # For removing all data containing the fourth digit
val_filtered_images = []
val_filtered_labels = []
test_filtered_images = []
test_filtered_labels = []

for i, lab in enumerate(mnist.validation.labels):
    if np.argmax(lab) in selected_labels:
        val_filtered_images.append(mnist.validation.images[i])
        val_filtered_labels.append(np.delete(mnist.validation.labels[i], 3))
for j, lab in enumerate(mnist.test.labels):
    if np.argmax(lab) in selected_labels:
        test_filtered_images.append(mnist.test.images[j])
        test_filtered_labels.append(np.delete(mnist.test.labels[j], 3))

# Convolutional layer
x_image = tf.reshape(x, [-1, 28, 28, 1])
W_conv = tf.Variable(tf.truncated_normal([5, 5, 1, 30], stddev=0.1))
b_conv = tf.Variable(tf.constant(0.1, shape=[30]))
h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([12 * 12 * 30, 500], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[500]))
h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

# Output layer
# : Change number of output categories from 10 to 9
W_fc2 = tf.Variable(tf.truncated_normal([500, 9], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[9]))
y_hat = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = - tf.reduce_sum(Y_before * tf.log(y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y_before, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print("=Before Transferring=============")
    print("=================================")
    print("|Epoch\tBatch\t|Train\t|Val\t|")
    print("|===============================|")
    for j in range(5):
        for i in range(550):
            batch = mnist.train.next_batch(100)

            # Filtered dataset exclude fourth label
            batch_filtered_images = []
            batch_filtered_labels = []
            for k, lab in enumerate(batch[1]):
                if np.argmax(lab) in selected_labels:
                    batch_filtered_images.append(batch[0][k])
                    batch_filtered_labels.append(np.delete(batch[1][k], 3))

            train_step.run(feed_dict={x: batch_filtered_images, Y_before: batch_filtered_labels})
            if i % 110 == 109:
                train_accuracy = accuracy.eval(feed_dict={x: batch_filtered_images, Y_before: batch_filtered_labels})
                val_accuracy = accuracy.eval(feed_dict={x: val_filtered_images, Y_before: val_filtered_labels})
                print("|%d\t|%d\t|%.4f\t|%.4f\t|" % (j+1, i+1, train_accuracy, val_accuracy))
    print("|===============================|")
    test_accuracy = accuracy.eval(feed_dict={x: test_filtered_images, Y_before: test_filtered_labels})
    print("test accuracy=%.4f" % (test_accuracy))

    # Define new parameters from last CNN to train transfer learning
    W_conv_trans = sess.run(W_conv)
    b_conv_trans = sess.run(b_conv)
    W_fc1_trans = sess.run(W_fc1)
    b_fc1_trans = sess.run(b_fc1)
    W_fc2_temp = sess.run(W_fc2)
    b_fc2_temp = sess.run(b_fc2)

Y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Convolutional layer for transfer learning
h_conv = tf.nn.conv2d(x_image, W_conv_trans, strides=[1, 1, 1, 1], padding='VALID')
h_relu = tf.nn.relu(h_conv + b_conv_trans)
h_pool = tf.nn.max_pool(h_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully-connected layer for transfer learning
h_pool_flat = tf.reshape(h_pool, [-1, 12*12*30])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1_trans) + b_fc1_trans)

# Output layer of transfer learning
# W_fc2 = tf.Variable(np.c_[W_fc2_temp, np.full((500, 1), 0.1)])
# b_fc2 = tf.Variable(np.append(b_fc2_temp, 0.1))
W_fc2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_hat = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model for transfer learning
cross_entropy = - tf.reduce_sum(Y_ * tf.log(y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print("=Transfer learning===============")
    print("=================================")
    print("|Epoch\tBatch\t|Train\t|Val\t|")
    print("|===============================|")
    for j in range(5):
        for i in range(550):
            batch = mnist.train.next_batch(100)
            train_step.run(feed_dict={x: batch[0], Y_: batch[1]})
            if i % 50 == 49:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], Y_: batch[1]})
                val_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, Y_: mnist.validation.labels})
                print("|%d\t|%d\t|%.4f\t|%.4f\t|" % (j+1, i+1, train_accuracy, val_accuracy))
    print("|===============================|")
    test_accuracy_trans = accuracy.eval(feed_dict={x: mnist.test.images, Y_: mnist.test.labels})
    print("test accuracy=%.4f" % (test_accuracy_trans))