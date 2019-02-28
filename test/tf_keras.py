import tensorflow as tf
from tensorflow import keras as k
import numpy as np

# Sequential
model = k.Sequential()
model.add(k.layers.Dense(64,activation='relu'))
model.add(k.layers.Dense(64,activation='relu'))
model.add(k.layers.Dense(10,activation='softmax'))
model.compile(optimizer=tf.train.AdamOptimizer(1e-3),loss=k.losses.categorical_crossentropy,metrics=[k.metrics.mse])

data = np.random.random((1000,32))
labels = np.random.random((1000,10))

val_data = np.random.random((100,32))
val_labels = np.random.random((100,10))

model.fit(data,labels,epochs=10,batch_size=32,validation_data=(val_data,val_labels))

#Model
inputs =k.Input(shape=(32,))
x = k.layers.Dense(64,activation='relu')(inputs)
x = k.layers.Dense(64,activation='relu')(x)

pre = k.layers.Dense(10,activation='softmax')(x)

model = k.Model(inputs = inputs,outputs =pre)
model.compile(optimizer=tf.train.AdamOptimizer(1e-3),loss=k.losses.categorical_crossentropy,metrics=[k.metrics.mse])

model.fit(data,labels,epochs=10,batch_size=32,validation_data=(val_data,val_labels))