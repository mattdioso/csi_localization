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
from sklearn.metrics import classification_report
np.set_printoptions(threshold=sys.maxsize, suppress=False)
octave.addpath('../../matlab/')

def preprocess_csi(csi):
  csi = csi.astype(complex)
  csi_b = np.linalg.norm(csi)
  csi_norm = csi/csi_b

  mags = []
  phases = []
  for x in csi:
    A = x.real
    B = x.imag

    square = np.square(A) + np.square(B)
    mag = np.sqrt(square)
    phase = np.arctan(B/A)

    phases.append(phase)
    mags.append(mag)

  np_mags = np.array(mags)
  return np_mags

def clean_image(csi):
  result_arr = np.empty([1, 998])
  for i in range(csi.shape[0]):
    np_arr = csi[i][csi[i] != 0]
    try:
      result_arr = np.vstack((result_arr, np_arr))
    except:
      np_arr_shape = np_arr.shape[0]
      diff = 998 - np_arr_shape
      for i in range(diff):
        np_arr = np.append(np_arr ,0)
      result_arr = np.vstack((result_arr, np_arr))
  result_arr = np.delete(result_arr, 0, axis=0)
  return result_arr

def np_csi_normalized(csi):
  csi_norm = (csi - np.min(csi))/(np.max(csi) - np.min(csi))
  return csi_norm

def extract_csi(file):
  mat_file = file
#  info_arr = np.array([env_id, L, W, H, grid, antenna, los, num_trace])
  try:
    alldata = octave.extract_function(mat_file)
    # [0][i for num packets][0][0]['nss'][0]
  except Exception as e:
    print(e)
    alldata=""

  if not isinstance(alldata, str):
    result_arr = np.empty([1, 1032])
    #print(info_arr)
    #print(alldata[0][0][0][0][3]['nss'][0]['data'].shape)
    try:
      csi = alldata[0][0][0][0]['nss'][0]['data']
    except Exception as e:
      #print(e)
      try:
        print("failed nss 0, trying 3")
        csi = alldata[0][0][0][0][3]['nss'][0]['data']
      except:
        try:
          print("failed nss 3, trying 2")
          csi = alldata[0][0][0][0][2]['nss'][0]['data']
        except:
          print("failed nss 2, trying 1")
          csi = alldata[0][0][0][0][1]['nss'][0]['data']
          #print(csi)
      #arr_out = np.concatenate((info_arr, csi), axis=None)
      #result_arr = np.vstack((result_arr, arr_out))
      #print(arr_out)
    return csi
    #result_arr = np.delete(result_arr, 0, axis=0)
    #print(result_arr.shape)
    #csv_file_name = file[:-3] + "csv"
    #print(csv_file_name)
    #pd.DataFrame(result_arr).to_csv(os.path.join(csv_file_dir, csv_file_name), index=False, header=None)
    count += 1
  else:
    print("alldata is empty")
    return []

test_dir = "/home/matt/hidden-device-research/results/tests/batch_1709/mats/"
files = ["capture_1_trace.mat", "capture_2_trace.mat", "capture_4_trace.mat", "capture_8_trace.mat"]
info_arr = [[7.3], [4.1], [2.7] ,[0.044]]
model = keras.models.load_model('cnn_model_equal_augmented_norm_98F1.h5', custom_objects={'leaky-relu': LeakyReLU()})
start = timeit.default_timer()
result_arr = np.empty([1, 1024])
for file in files:
#  print(file)
  csi = preprocess_csi(extract_csi(test_dir + file))
  print(csi.shape)
"""
  csi = np.array(np_csi_normalized(csi))
  result_arr = np.vstack((result_arr, csi))
#  print(csi)
result_arr = np.delete(result_arr, 0, axis=0)
result_arr = np.hstack((info_arr, result_arr))
result_arr = clean_image(result_arr)[:, :-1]
result_arr = tf.expand_dims(result_arr, axis=0)

test_arr = np.vstack((result_arr, result_arr, result_arr))
print(test_arr.shape)

X, Y = load_unseen_data()
Y -= 1
#print(np.array(test_arr))
prediction = model.predict(np.array(test_arr))
print(np.argmax(prediction, axis=1))

#Y_pred = model.predict(X, batch_size=64, verbose=1)
#Y_pred_bool = np.argmax(Y_pred, axis=1)
#print(classification_report(Y, Y_pred_bool))

stop = timeit.default_timer()
print("runtim: " + str(stop-start))
"""
