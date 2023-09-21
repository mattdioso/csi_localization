#!/usr/bin/env python3
from tensorflow.keras.layers import Input, Dense, Activation, LeakyReLU
from tensorflow.keras.layers import LayerNormalization, Layer, Conv1D, Conv2D
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Flatten, MaxPooling2D
from tensorflow.keras import utils
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.saving import get_custom_objects

from sklearn.metrics import classification_report, confusion_matrix
from load_data import load_data, load_unseen_data, load_equal_data
import time

def build_model(input_shape):
  #inputs = Input(shape=input_shape)
  model = Sequential()
#  model.add(Input(shape=input_shape))
  model.add(Conv2D(filters=128, kernel_size=1, activation="relu"))
  model.add(Dropout(rate=0.1))
  model.add(Flatten())
  model.add(Dense(units=128, activation="relu"))
  model.add(Dense(units=21, activation="softmax"))
  opt = keras.optimizers.Adam(learning_rate=0.001)
  #model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model

NAME = 'test-cnn-100epoch-batch16-{}_nonnormalized'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
get_custom_objects().update({'leaky-relu': Activation(LeakyReLU())})

"""
conv2d 256, 1, leaky-relu
conv2d 256, 1, leaky-relu
conv2d 256, 1, leaky-relu
dense 128, leaky-relu
dense 21, leaky-relu
"""

X_train, Y_train, x_test, y_test = load_equal_data(norm=False)
Y_train -= 1
y_test -= 1
X_train = tf.expand_dims(X_train, axis=-1)
input_shape = X_train.shape[1:]
print(input_shape)
#model = build_model(input_shape)
model = Sequential()
model.add(Input(shape=input_shape))
model.add(Conv2D(filters=512, kernel_size=1))#, activation="leaky-relu"))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=256, kernel_size=1))#, activation="leaky-relu"))
model.add(LeakyReLU())
model.add(Conv2D(filters=256, kernel_size=1))#, activation="leaky-relu"))
model.add(LeakyReLU())
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(units=128))#, activation="leaky-relu"))
model.add(LeakyReLU())
model.add(Dense(units=21, activation="softmax"))
opt = keras.optimizers.Adam()
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, epochs=10, validation_data=[x_test, y_test], batch_size=64, callbacks=[tensorboard])
val_loss, val_acc = model.evaluate(x_test, y_test)
print("loss: " + str(val_loss))
print("accuracy: " + str(val_acc))

X_unseen, Y_unseen = load_unseen_data(norm=False)
Y_unseen -= 1
Y_pred = model.predict(X_unseen, batch_size=64, verbose=1)
y_pred_bool = np.argmax(Y_pred, axis=1)

print(classification_report(Y_unseen, y_pred_bool))

c_matrix = confusion_matrix(Y_unseen, y_pred_bool, normalize='pred')

plt.imsave('c_matrix22.png', c_matrix)
model.save('cnn_model_equal_augmented_fixed', save_format='h5')
