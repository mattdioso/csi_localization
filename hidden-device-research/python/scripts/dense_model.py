#!/usr/bin/env python3
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import LayerNormalization, Layer, Conv1D, Conv2D
from tensorflow.keras.layers import Dropout, GlobalAveragePooling1D, Flatten
from tensorflow.keras import utils
from tensorflow.keras.models import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from load_data import load_data
import time

def build_model(input_shape):
  #inputs = Input(shape=input_shape)
  model = Sequential()
#  model.add(Input(shape=input_shape))
  model.add(Conv2D(filters=128, kernel_size=1, activation="relu"))
  model.add(Conv2D(filters=256, kernel_size=2, activation="relu"))
  model.add(Conv2D(filters=512, kernel_size=2, activation="relu"))
  model.add(Conv2D(filters=256, kernel_size=2, activation="relu"))
  model.add(Dropout(rate=0.3))
  model.add(Flatten())
  model.add(Dense(units=2048, activation="relu"))
  model.add(Dense(units=21, activation="softmax"))
  opt = keras.optimizers.Adam(learning_rate=0.001)
  #model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model

NAME = 'test-dense-100epoch-batch16-{}_nonnormalized'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

X_train, Y_train, x_test, y_test = load_data()
Y_train -= 1
y_test -= 1
X_train = tf.expand_dims(X_train, axis=-1)
input_shape = X_train.shape[1:]
print(input_shape)
#model = build_model(input_shape)
model = Sequential()
model.add(Input(shape=input_shape))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
#model.add(Dropout(0.1))
model.add(Dense(512, activation="relu"))
#model.add(Dropout(0.1))
model.add(Dense(256, activation="relu"))
#model.add(Dropout(0.1))
model.add(Dense(128, activation="relu"))
#model.add(Dense(64, activation="relu"))
#model.add(Dense(32, activation="relu"))
#model.add(Dropout(0.1))
model.add(Dense(21, activation="softmax"))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.build(input_shape)
model.summary()
model.fit(X_train, Y_train, epochs=5, validation_data=[x_test, y_test], callbacks=[tensorboard])
val_loss, val_acc = model.evaluate(x_test, y_test)
print("loss: " + str(val_loss))
print("accuracy: " + str(val_acc))
model.save('dense_model', save_format='h5')
