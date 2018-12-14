import tensorflow as tf
import numpy as np


with tf.Session() as sess:
    # load graph to Saver object
    new_saver = tf.train.import_meta_graph('./my-model-1000.meta')

    # read values of the model, to sess
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    # current graph is the loaded graph
    graph = tf.get_default_graph()

    # get variables and op from the loaded graph
    input = graph.get_tensor_by_name("input:0")
    v1 = graph.get_tensor_by_name("op_v1:0")
    v2 = graph.get_tensor_by_name("op_v2:0")

    weight = sess.run("weight1:0")
    bias = sess.run("bias1:0")
    print("weight1 restored from model:\nshape={0}\nvalue=\n{1}\n".format(weight.shape, weight))
    print("bias1 restored from model:\nshape={0}\nvalue=\n{1}\n".format(bias.shape, bias))

    input_val = np.array([[1, 2.0], [3, 4], [5, 6]])
    print("input_val:\n{}\n".format(input_val))

    v1, v2 = sess.run([v1, v2], feed_dict={input: input_val})
    print("op v1:\n{}\n".format(v1))
    print("op v2:\n{}\n".format(v2))

    expected = np.matmul(input_val, weight) + bias
    print("expected:\n{}\n".format(expected))
    assert np.allclose(expected, v2)

    weight2 = sess.run("weight2:0")
    print("* weight2 restored from model:\n{}".format(weight2))
