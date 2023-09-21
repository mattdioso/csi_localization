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
import matplotlib.pyplot as plt
from calculate_distance_error import calculate_mean_error, calculate_distance_error
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
    test_file = "/Volumes/External Drive/research/mscse-capstone/hidden-device-research/datasets/mat_files/1709/1709_6.7_3.6_2.5_10_2_los_10_trace.mat"
    alldata = octave.extract_function(mat_file)
    # [0][i for num packets][0][0]['nss'][0]
  except Exception as e:
#    print(e)
    alldata=""

  if not isinstance(alldata, str):
    result_arr = np.empty([1, 1032])
    #print(info_arr)
    #print(alldata[0][0][0][0][3]['nss'][0]['data'].shape)
    try:
      csi = alldata['core'][0][3]['nss'][0]['data'][0]
    except Exception as e:
      #print(e)
      try:
 #       print("failed nss 0, trying 3")
        #print(alldata['core'][0][2]['nss'][0]['data'][0])
        csi = alldata['core'][0][2]['nss'][0]['data'][0]
      except:
        try:
#          print("failed nss 3, trying 2")
          #csi = alldata[0][0][0][0][2]['nss'][0]['data']
#          print(alldata['core'][0]['nss'][0]['data'])
          csi = alldata['core'][0][1]['nss'][0]['data'][0]
        except:
#          print("failed nss 2, trying 1")
          #csi = alldata[0][0][0][0][1]['nss'][0]['data']
          csi = alldata['core'][0]['nss'][0]['data'][0]
      #print(csi)
#    arr_out = np.concatenate((info_arr, csi), axis=None)
    return csi
  else:
    print("alldata is empty")
    return []


dense_dir = "/home/matt/hidden-device-research/results/tests/bedroom/dense/mats/"
undense_dir = "/home/matt/hidden-device-research/results/tests/bedroom/undense/mats/"



info_arr = [[10.6], [10.6], [3.3] ,[7.35]]
home_info_arr = [[6.9], [4.0], [2.1], [14.7]]
css_info_arr = [[8.6], [5.2], [3.6], [8.09]]
bedroom_dense_info = [[3.3], [4.5], [2.7], [1.5]]
model = keras.models.load_model('cnn_model_equal_augmented_norm_fixed_98F1.h5', custom_objects={'leaky-relu': LeakyReLU()})
start = timeit.default_timer()
distance_wrong = 0
num_wrong = 0
files = {'1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': [], '11': [], '12': [], '13': [], '14': [], '15': [], '16': [], '17': [], '18': [], '19':[], '20': [], '21': []}
for file in sorted(os.listdir(dense_dir)):
    #print(file)
    ans = file.split("_")[1]
    files[ans].append(file)
    '''
    result_arr = np.empty([1, 1024])
    for file in files:
      print(file)
      csi = preprocess_csi(extract_csi(test_dir + file))
      csi = np.array(np_csi_normalized(csi))
      result_arr = np.vstack((result_arr, csi))
    result_arr = np.delete(result_arr, 0, axis=0)
    result_arr = np.hstack((info_arr, result_arr))
    result_arr = clean_image(result_arr)[:, :-1]
    result_arr = tf.expand_dims(result_arr, axis=0)
    prediction = model.predict(np.array(result_arr))
    Y_true = ans
    if int(np.argmax(prediction, axis=1)[0]) != int(ans):
        print(str(np.argmax(prediction, axis=1)[0]) + "\t" + str(ans))
        num_wrong += 1
    else:
        print(str(np.argmax(prediction, axis=1)[0]) + "\t" + str(ans))
    distance_wrong += calculate_distance_error(np.argmax(prediction, axis=1)[0], Y_true)
    '''
#print(files)
Y_true = []
Y_pred = []
for obj in files:
    ans = obj
    mats = files[obj]
    result_arr = np.empty([1, 1024])
    for file in mats:
        csi = preprocess_csi(extract_csi(dense_dir + file))
        csi = np.array(np_csi_normalized(csi))
        result_arr = np.vstack((result_arr, csi))
    result_arr = np.delete(result_arr, 0, axis=0)
    result_arr = np.hstack((bedroom_dense_info, result_arr))
    result_arr = clean_image(result_arr)[:, :-1]
    result_arr = tf.expand_dims(result_arr, axis=0)
    prediction = model.predict(np.array(result_arr))
    Y_true.append(int(ans))
    Y_pred.append(int(np.argmax(prediction, axis=1)[0]))
    if int(np.argmax(prediction, axis=1)[0]) != int(ans):
        num_wrong += 1
    distance_wrong += calculate_distance_error(np.argmax(prediction, axis=1)[0], int(ans))
for i in range(len(Y_true)):
    Y_true[i] -= 1
print(Y_true)
print(Y_pred)
#print(calculate_mean_error(Y_pred, Y_true))
print(distance_wrong/21)

c_matrix = confusion_matrix(Y_true, Y_pred, normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
disp.plot()
plt.savefig('confusion_small_matrix.png')

stop = timeit.default_timer()
print("runtim: " + str(stop-start))
