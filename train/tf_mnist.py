from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.keras import layers
import os
import numpy as np
os.chdir('/home/zzy/PycharmProjects/tf/')

data = input_data.read_data_sets('data/MNIST/',one_hot=True)

print(data.train.labels[0])
input = layers.Input(shape=(28,28,1))

layer_a = layers.Conv2D(16,(5,5),activation='relu')(input)
layer_a1 = layers.MaxPool2D((2,2),2)(layer_a)
layer_a2 = layers.Dropout(0.25)(layer_a1)

layer_b = layers.Conv2D(36,(5,5),activation='relu')(layer_a2)
layer_b1 = layers.MaxPool2D((2,2),2)(layer_b)
layer_b2 = layers.Dropout(0.25)(layer_b1)
layer_b3 = layers.Flatten()(layer_b2)
layer_c = layers.Dense(128,activation='relu')(layer_b3)
layer_c1 = layers.Dropout(0.25)(layer_c)

output = layers.Dense(10,activation='softmax')(layer_c1)

model = tf.keras.Model(input,output)

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True),metrics = ['accuracy'])

train_images = np.reshape(data.train.images,(-1,28,28,1))
var_images = np.reshape(data.validation.images,(-1,28,28,1))
model.fit(train_images,data.train.labels, epochs=500, batch_size=64, validation_data=(var_images, data.validation.labels))


