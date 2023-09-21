#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os 
import sys
from oct2py import octave
import timeit
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.saving import get_custom_objects
from tensorflow.keras.layers import Input, Dense, Activation, LeakyReLU
#get_custom_objects().update({'leaky-relu': Activation(LeakyReLU())})
from load_data import load_unseen_data
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from calculate_distance_error import calculate_mean_error
np.set_printoptions(threshold=sys.maxsize, suppress=False)
import matplotlib.pyplot as plt

model = keras.models.load_model('cnn_model_equal_augmented_norm_fixed_98F1.h5', custom_objects={'leaky-relu': LeakyReLU()})

X_unseen, Y_unseen = load_unseen_data(norm=True)
Y_unseen -= 1
Y_pred = model.predict(X_unseen, verbose=1)
y_pred_bool = np.argmax(Y_pred, axis=1)
model.summary()
print(classification_report(Y_unseen, y_pred_bool))

grids  = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0, '19':0, '20': 0, '21': 0}

cf_wrong = []
cf_right = []
for ans, pred in zip(Y_unseen, Y_pred):
  if int(ans) != int(np.argmax(pred, axis=0)):
  #print(str(int(ans+1)))
  #if grids[str(int(ans+1))] != 25:
    cf_wrong.append(int(np.argmax(pred, axis=0)))
    cf_right.append(ans)
    grids[str(int(ans+1))] += 1
print(grids)
for ans, pred in zip(Y_unseen, Y_pred):
  if grids[str(int(ans+1))] != 100:
    cf_wrong.append(int(np.argmax(pred, axis=0)))
    cf_right.append(ans)
    grids[str(int(ans+1))] += 1

print(grids)
c_matrix = confusion_matrix(cf_right, cf_wrong)
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
disp.plot(include_values=False, cmap='cividis')
'''
c_matrix = confusion_matrix(Y_unseen, y_pred_bool, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
disp.plot(include_values=False)
'''

#disp.figure_.savefig('confusion_matrix.png')

#print(calculate_mean_error(Y_pred, Y_unseen))

#plt.imsave('confusion_matrix.png', disp.plot())
plt.savefig('confusion_matrix.png')
