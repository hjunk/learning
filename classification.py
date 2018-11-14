import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# Hyperparameters
training_size = 500         # Size of training data set
training_epochs = 200       # Number of iterations
val_size = 250              # Size of validation data set
test_size = 250             # Size of test data set
nh = 20                     # Number of neurons in hidden layer
lr = 0.1                    # learning rate for gradient descent algorithm
var_init = 0.1              # Standard deviation of initializer

# Training Data
r = np.random.normal(0, 1, training_size)
t = np.random.uniform(0, 2 * np.pi, training_size)

x_data = []
y_data = []
x_1_unlabeled = []
x_1_labeled = []
x_2_unlabeled = []
x_2_labeled = []

for i in range(training_size):
    if (i % 2 == 0):
        x_data.append([r[i] * np.cos(t[i]), r[i] * np.sin(t[i])])
        y_data.append([0])
        x_1_unlabeled.append(r[i] * np.cos(t[i]))
        x_2_unlabeled.append(r[i] * np.sin(t[i]))
    else:
        x_data.append([(r[i] + 5) * np.cos(t[i]), (r[i] + 5) * np.sin(t[i])])
        y_data.append([1])
        x_1_labeled.append((r[i] + 5) * np.cos(t[i]))
        x_2_labeled.append((r[i] + 5) * np.sin(t[i]))

x_ = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Plot
plt.scatter(x_1_labeled, x_2_labeled, facecolor='green', edgecolor="green", s=20, label="Classifier output ~ 1")
plt.scatter(x_1_unlabeled, x_2_unlabeled, facecolor='blue', edgecolor="blue", s=20, label="Classifier output ~ 0")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.title("EE488 Project#1 20080242 Hyuk Jun Kim")
plt.legend(loc="best")
plt.savefig('plt_task1.png')

# Validation Data & test data
r_val = np.random.normal(0, 1, val_size)
t_val = np.random.uniform(0, 2 * np.pi, val_size)
r_test = np.random.normal(0, 1, test_size)
t_test = np.random.uniform(0, 2 * np.pi, test_size)

x_val = []
y_val = []
x_test = []
y_test = []

for i in range(val_size):
    if (i % 2 == 0):
        x_val.append([r_val[i] * np.cos(t_val[i]), r_val[i] * np.sin(t_val[i])])
        y_val.append([0])
        x_test.append([r_test[i] * np.cos(t_test[i]), r_test[i] * np.sin(t_test[i])])
        y_test.append([0])
    else:
        x_val.append([(r_val[i] + 5) * np.cos(t_val[i]), (r_val[i] + 5) * np.sin(t_val[i])])
        y_val.append([1])
        x_test.append([(r_test[i] + 5) * np.cos(t_test[i]), (r_test[i] + 5) * np.sin(t_test[i])])
        y_test.append([1])

# Fully-connected layer
W_fc1 = tf.Variable(tf.truncated_normal([2, nh], stddev=var_init))
b_fc1 = tf.Variable(tf.truncated_normal([nh], stddev=var_init))
h_fc1 = tf.nn.relu(tf.matmul(x_, W_fc1) + b_fc1)

# Output layer
W_fc2 = tf.Variable(tf.truncated_normal([nh, 1], stddev=var_init))
b_fc2 = tf.Variable(tf.truncated_normal([1], stddev=var_init))
y_hat = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Train and Evaluate the Model
cross_entropy = - tf.reduce_mean(y_ * tf.log(y_hat) + (1 - y_) * tf.log(1 - y_hat))
train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.cast(y_hat > 0.5, dtype=tf.float32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print("=================================")
print("|Epoch\t|Train\t|Val\t|")
print("|===============================|")
for i in range(training_epochs):
    val = sess.run([train_step, cross_entropy], feed_dict={x_: x_data, y_: y_data})
    if i % 10 == 9:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x_: x_data, y_: y_data})
        val_accuracy = accuracy.eval(session=sess, feed_dict={x_: x_val, y_: y_val})
        print("|%d\t|%.4f\t|%.4f\t|" % (i + 1, train_accuracy, val_accuracy))
print("|===============================|")
test_accuracy = accuracy.eval(session=sess, feed_dict={x_: x_test, y_: y_test})
print("test accuracy=%.4f" % (test_accuracy))