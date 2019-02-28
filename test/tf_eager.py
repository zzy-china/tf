import tensorflow as tf



tf.enable_eager_execution()
x = [[2]]
m = tf.matmul(x,x)
print(" x*x = {}".format(m))

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

model = MyModel()
