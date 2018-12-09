from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# ----------- PARAM -----------
size = 50
num_iteration = 10
# ----------- DATA ------------
train_x = np.random.uniform(-1, 1, size)
# randn: standard_normal distribution, u=0, s=1
train_y = 3 * train_x + np.random.randn(size) * 0.33

# ----------- GRAPH ------------

# only one sample, scalar x and scalar y
x = tf.placeholder("float", name="x")
y = tf.placeholder("float", name="y")

w = tf.Variable(0.0, name="weights")
y_model = tf.multiply(x, w)

cost = tf.square(y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# init should be after all variable definitions
init = tf.global_variables_initializer()
# for op in tf.get_default_graph().get_operations():
#     print(op.name)

# ----------- SUMMARY ------------
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# ----------- RUN ----------------
# 若summary后面没有sess.run, summary就不会写出event文件
with tf.Session() as sess:

    sess.run(init)
    for i in range(num_iteration):
        for (x_, y_) in zip(train_x, train_y):
            sess.run(train_op, feed_dict={x: x_, y: y_})

    print(sess.run(w))

# cd 1.first-tutorial/
# tensorboard --logdir .
# (open link)

