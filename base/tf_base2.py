import tensorflow as tf

x = tf.ones([3,2])
y = tf.ones([1,2])

with tf.Session() as sess:
    print(sess.run(x+y))