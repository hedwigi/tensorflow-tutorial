import tensorflow as tf
import numpy as np

a = tf.constant(3.0, dtype=tf.float32, name='a')
b = tf.constant(4.0)
total = a + b

print(a)
print(b)
print(total)

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2

# ----------

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

# ----------

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()

print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1,2], y: [4,2]}))

val = sess.run(total)
print(val)

dic = sess.run({'ab': (a, b), 'total': total})
print(dic)

print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))



