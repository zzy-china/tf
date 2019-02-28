import tensorflow as tf
import os

os.chdir('/home/zzy/PycharmProjects/Test/')

# create Sesssion
sess = tf.Session()

# const
const1 = tf.constant(3.0, dtype=tf.float32)
const2 = tf.constant(4.0)

# tensor 0~n --- rank
var0 = tf.Variable(1.0, tf.float32)  # []
var1 = tf.Variable([1.0, 2.0], tf.float32)  # [2]
var2 = tf.Variable([[1, 2], [3, 4]], tf.int16)  # [2,2]
varn = tf.zeros([100, 288, 299, 3])

# function eg. shape reshape cast

# variable
variable_a =tf.get_variable('a',[1,2,3])
variable_b =tf.get_variable('b',[1,2,3],dtype=tf.float32,initializer=tf.zeros_initializer())
variable_c =tf.get_variable('c',dtype=tf.float32,initializer=tf.constant([23.0,32.0]))

# device-type-variable
with tf.device("/device:GPU:0"):
    v =tf.get_variable("v",[1])

# variable_scope

# placeholder
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

# feed_dict
print(sess.run(z, feed_dict={x: 3, y: 4}))

# Layer
layer_x = tf.placeholder(tf.float32, shape=[None, 3])  # (n,3)

# Dense
liner_model = tf.layers.Dense(units=1)
layer_y = liner_model(layer_x)
# or y = tf.layers.Dense(x,units=1)

# init
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(layer_y, feed_dict={layer_x: [[1, 2, 3]]}))

# features
features = {
    'sales': [[5], [10], [8], [9]],
    'department': ['s', 's', 'm', 'm']
}
department_column = tf.feature_column.categorical_column_with_vocabulary_list('department', ['s', 'm'])
department_column = tf.feature_column.indicator_column(department_column)
columns = [tf.feature_column.numeric_column('sales'), department_column]
inputs = tf.feature_column.input_layer(features, columns)

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()

sess = tf.Session()
sess.run((var_init, table_init))

print(sess.run(inputs))
# result
# [[ 1.  0.  5.]
#  [ 1.  0. 10.]
#  [ 0.  1.  8.]
#  [ 0.  1.  9.]]

# save Variable

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)
    saver.save(sess,"data/model.ckpt")

    saver.restore('sess','data/model.ckpt')
