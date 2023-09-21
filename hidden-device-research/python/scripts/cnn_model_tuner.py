#!/usr/bin/env python3
from tensorflow.keras.layers import Input, Dense, Activation, LeakyReLU, MaxPooling2D
from tensorflow.keras.layers import LayerNormalization, Layer, Conv1D, Conv2D
from tensorflow.keras.layers import Dropout, GlobalAveragePooling1D, Flatten
from tensorflow.keras import utils
from tensorflow.keras.models import Model, Sequential
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.saving import get_custom_objects

from load_data import load_data
import time

LOG_DIR = f"{int(time.time())}"
get_custom_objects().update({'leaky-relu': Activation(LeakyReLU())})
NAME = 'test-cnn-100epoch-batch16-{}_nonnormalized'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))


X_train, Y_train, x_test, y_test = load_data()
Y_train -= 1
y_test -= 1
X_train = tf.expand_dims(X_train, axis=-1)
input_shape = X_train.shape[1:]

def build_model(hp):
  
  hp_activation = hp.Choice('activation', values=['relu', 'leaky-relu'])
  hp_conv_filters = hp.Int('conv_filters', min_value=256, max_value=512, step=256)
  hp_conv_kernel = hp.Int('conv_kernel', min_value=1, max_value=2, step=1)
  hp_dense_units = hp.Int('dense_units', min_value=128, max_value=256, step=128)
  hp_dropout = hp.Float('dropout_units', min_value=0.0, max_value=0.2, step=0.1)

  model = Sequential()
  model.add(Input(shape=input_shape))
  model.add(Conv2D(filters=hp_conv_filters, kernel_size=hp_conv_kernel, activation=hp_activation))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(rate=hp_dropout))
  model.add(Conv2D(filters=hp_conv_filters, kernel_size=hp_conv_kernel, activation=hp_activation))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(filters=hp_conv_filters, kernel_size=hp_conv_kernel, activation=hp_activation))
  model.add(Dropout(rate=hp_dropout))
  model.add(Flatten())
  model.add(Dense(units=hp_dense_units, activation=hp_activation))
  model.add(Dense(units=21, activation="softmax"))
  opt = keras.optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=3, directory=LOG_DIR)
tuner.search(x=X_train, y=Y_train, verbose=2, epochs=5, batch_size=16, callbacks=[tensorboard, stop_early], validation_data=(x_test, y_test))
tuner.search_space_summary()

print("BEST HYPERPARAMETERS: ")
print(tuner.get_best_hyperparameters()[0].values)

print("RESULTS SUMMARY: ")
print(tuner.results_summary())

print("BEST MODEL SUMMARY: ")
print(tuner.get_best_models()[0].summary())

#model = build_model()
#model.summary()
#model.fit(X_train, Y_train, epochs=5, validation_data=[x_test, y_test], batch_size=16, callbacks=[tensorboard])
#val_loss, val_acc = model.evaluate(x_test, y_test)
#print("loss: " + str(val_loss))
#print("accuracy: " + str(val_acc))
