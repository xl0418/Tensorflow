"""
Simple exercises to get used to TensorFlow API
You should thoroughly test your code.
TensorFlow's official documentation should be your best friend here
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

sess = tf.InteractiveSession()
###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda: x + y, lambda: x - y)
print(sess.run(out))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

x = tf.random_uniform([],minval=-1,maxval=1)  # Empty array as shape creates a scalar.
y = tf.random_uniform([],minval=-1,maxval=1)
out = tf.cond(tf.greater(y, x), lambda: x + y, lambda: x - y)
print(sess.run(out))

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]]
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

# Solution 1
a = tf.constant([[0, -2, -1], [0, 1, 2]],tf.float32)
b = tf.constant([[0, 0, 0], [0, 0, 0]],tf.float32)

c = tf.equal(a,b)
print(sess.run(c))

# Solution 2
a = tf.placeholder(tf.float32,shape=[2,3])
b = tf.placeholder(tf.float32,shape=[2,3])
c = tf.equal(a,b)
print(sess.run(c,feed_dict={a:[[0, -2, -1], [0, 1, 2]],b:[[0, 0, 0], [0, 0, 0]]}))

###############################################################################
# 1d: Create the tensor x of value
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

a = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
 30.97266006,  26.67541885,  38.08450317,  20.74983215,
 34.94445419,  34.45999146,  29.06485367,  36.01657104,
 27.88236427,  20.56035233,  30.20379066,  29.51215172,
 33.71149445,  28.59134293,  36.05556488,  28.66994858],tf.float32)
greaterthan30_index = tf.where(tf.greater(a,30))
gt30 = tf.gather(a,greaterthan30_index)
print(sess.run(greaterthan30_index))
print(sess.run(gt30))

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

a = tf.diag(tf.range(1,7))
print(sess.run(a))

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

a = tf.random_normal(shape=[10,10],mean=0,stddev=1)
print(sess.run(a))
print(sess.run(tf.matrix_determinant(a)))



###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9],tf.float32)
uni_index = tf.unique(x)[0]
print(sess.run(uni_index))


###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.cond() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

x = tf.random_normal(shape=[300,1],mean=10,stddev=1)
y = tf.random_normal(shape=[300,1],mean=10,stddev=1)
sub = tf.reduce_mean(x - y)
print(sess.run(sub))
positive_result = tf.reduce_sum(tf.abs(x-y))
negative_result = tf.losses.mean_squared_error(x,y)
print(sess.run(negative_result))
con = tf.cond(tf.greater(sub,0.),lambda:positive_result,lambda:negative_result)

print(sess.run(con))


""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Linear regression
"""
import tensorflow as tf

import utils

DATA_FILE = "data/birth_life_2010.txt"

# Step 1: read in data from the .txt file
# data is a numpy array of shape (190, 2), each row is a datapoint
data, n_samples = utils.read_birth_life_data(DATA_FILE)

# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.get_variable('weights', initializer=tf.constant(0.0))
b = tf.get_variable('bias', initializer=tf.constant(0.0))

# Step 4: construct model to predict Y (life expectancy from birth rate)
Y_predicted = w * X + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    # Step 8: train the model
    for i in range(100):  # run 100 epochs
        for x, y in data:
            # Session runs train_op to minimize loss
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b])


""" Starter code for simple logistic regression model for MNIST
with tf.data module
MNIST dataset: yann.lecun.com/exdb/mnist/
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time



# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'c:/Liang/Tensorflow/MNIST_data'
# download_mnist(mnist_folder)
train, val, test = read_mnist(mnist_folder, flatten=True)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)  # if you want to shuffle your data
train_data = train_data.batch(batch_size)


# dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
# dataset = dataset.map(lambda x: x + 1) # 2.0, 3.0, 4.0, 5.0, 6.0
# dataset = dataset.shuffle(100)
# dataset = dataset.batch(5)
# dataset = dataset.repeat(2)
# iterator = dataset.make_one_shot_iterator()
# one_element = iterator.get_next()
# with tf.Session() as sess:
#     try:
#         while True:
#             print(sess.run(one_element))
#     except tf.errors.OutOfRangeError:
#         print("end!")

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(10000)  # if you want to shuffle your data
test_data = test_data.batch(batch_size)
#############################
########## TO DO ############
#############################

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)  # initializer for train_data
test_init = iterator.make_initializer(test_data)  # initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.get_variable(name='weights', shape=(784, 10), initializer=tf.random_normal_initializer())
b = tf.get_variable(name='bias', shape=(1, 10), initializer=tf.zeros_initializer())

#############################
########## TO DO ############
#############################


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = tf.matmul(img, w) + b
#############################
########## TO DO ############
#############################


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='loss')
loss = tf.reduce_mean(entropy)
#############################
########## TO DO ############
#############################


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#############################
########## TO DO ############
#############################


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs):
        sess.run(train_init)  # drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init)  # drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds / n_test))
writer.close()
