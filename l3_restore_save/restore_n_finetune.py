import tensorflow as tf


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph("./my-model-1000.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./"))

    # get the loaded graph
    graph = tf.get_default_graph()

    weight2 = graph.get_tensor_by_name("weight2:0")
    print("weight2 restored:\n{}\n".format(sess.run(weight2)))

    v3 = graph.get_tensor_by_name("op_v3:0")

    # It's an identity function
    v3 = tf.stop_gradient(v3)
    v3_shape = v3.get_shape().as_list()

    new_outputs = 4
    weights = tf.Variable(tf.truncated_normal([v3_shape[-1], new_outputs], stddev=0.05), name="weights")
    bias = tf.Variable(tf.constant(0.05, shape=[new_outputs]))
    output = tf.matmul(v3, weights) + bias
    pred = tf.nn.softmax(output)

    # re-init
    sess.run(tf.global_variables_initializer())

    weight2 = graph.get_tensor_by_name("weight2:0")
    print("weight2 after fine tune:\n{}\n".format(sess.run(weight2)))