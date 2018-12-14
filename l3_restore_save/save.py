import tensorflow as tf


# ---- GRAPH -----
input = tf.placeholder(tf.float32, shape=[None, 2], name="input")

# first fc layer
weight1 = tf.Variable(tf.random_normal(shape=[2, 4]), name="weight1")
# broadcast to every row
bias1 = tf.Variable(tf.random_normal(shape=[4]), name="bias1")
# matmul should carry out with two 2 dimensional matrices, [?, x], [x, y]
v1 = tf.matmul(input, weight1, name="op_v1")
v2 = tf.add(v1, bias1, name="op_v2")
v3 = tf.nn.relu(v2, name="op_v3")

# second fc layer
weight2 = tf.Variable(tf.random_normal(shape=[v3.get_shape().as_list()[-1], 3]), name="weight2")
bias2 = tf.Variable(tf.random_normal(shape=[3]), name="bias3")
v4 = tf.matmul(v3, weight2, name="op_v4")
v5 = tf.add(v4, bias2, name="op_v5")

# ---- SAVE -----
sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
saver.save(sess, './my-model', global_step=1000)
