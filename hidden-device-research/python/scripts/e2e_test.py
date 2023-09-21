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


test_dir = "/home/matt/hidden-device-research/results/tests/320/mats/"
files = ["capture_96in_N_1_trace.mat", "capture_96in_N_2_trace.mat", "capture_96in_N_4_trace.mat", "capture_96in_N_8_trace.mat"]
batches = [
    ["capture_116in_SE_1_trace.mat", "capture_116in_SE_2_trace.mat", "capture_116in_SE_4_trace.mat", "capture_116in_SE_8_trace.mat"],
    ["capture_124in_E_1_trace.mat", "capture_124in_E_2_trace.mat", "capture_124in_E_4_trace.mat", "capture_124in_E_8_trace.mat"],
    ["capture_135in_SW_1_trace.mat", "capture_135in_SW_2_trace.mat", "capture_135in_SW_4_trace.mat", "capture_135in_SW_8_trace.mat"],
    ["capture_156in_NE_1_trace.mat", "capture_156in_NE_2_trace.mat", "capture_156in_NE_4_trace.mat", "capture_156in_NE_8_trace.mat"],
    ["capture_16in_N_1_trace.mat", "capture_16in_N_2_trace.mat","capture_16in_N_4_trace.mat", "capture_16in_N_8_trace.mat"],
    ["capture_16in_S_1_trace.mat", "capture_16in_S_2_trace.mat", "capture_16in_S_4_trace.mat", "capture_16in_S_8_trace.mat"],
    ["capture_178in_S_1_trace.mat", "capture_178in_S_2_trace.mat", "capture_178in_S_4_trace.mat", "capture_178in_S_8_trace.mat"],
    ["capture_52in_NW_1_trace.mat", "capture_52in_NW_2_trace.mat", "capture_52in_NW_4_trace.mat", "capture_52in_NW_8_trace.mat"],
    ["capture_91in_S_1_trace.mat", "capture_91in_S_2_trace.mat", "capture_91in_S_4_trace.mat", "capture_91in_S_8_trace.mat"],
    ["capture_96in_N_1_trace.mat", "capture_96in_N_2_trace.mat", "capture_96in_N_4_trace.mat", "capture_96in_N_8_trace.mat"]
]

batches_ans = [6, 4, 14, 5, 8, 8, 9, 10, 9, 7]

css_batches = [
    ["capture_133in_NE_1_trace.mat", "capture_133in_NE_2_trace.mat", "capture_133in_NE_4_trace.mat", "capture_133in_NE_8_trace.mat"],
    ["capture_149in_NW_1_trace.mat", "capture_149in_NW_2_trace.mat", "capture_149in_NW_4_trace.mat", "capture_149in_NW_8_trace.mat"],
    ["capture_158in_E_1_trace.mat", "capture_158in_E_2_trace.mat", "capture_158in_E_4_trace.mat", "capture_158in_E_8_trace.mat"],
    ["capture_181in_SE_1_trace.mat", "capture_181in_SE_2_trace.mat", "capture_181in_SE_4_trace.mat", "capture_181in_SE_8_trace.mat"],
    ["capture_183in_SW_1_trace.mat", "capture_183in_SW_2_trace.mat","capture_183in_SW_4_trace.mat", "capture_183in_SW_8_trace.mat"],
    ["capture_33in_SE_1_trace.mat", "capture_33in_SE_2_trace.mat", "capture_33in_SE_4_trace.mat", "capture_33in_SE_8_trace.mat"],
    ["capture_42in_SW_1_trace.mat", "capture_42in_SW_2_trace.mat", "capture_42in_SW_4_trace.mat", "capture_42in_SW_8_trace.mat"],
    ["capture_61in_W_1_trace.mat", "capture_61in_W_2_trace.mat", "capture_61in_W_4_trace.mat", "capture_61in_W_8_trace.mat"],
    ["capture_78in_NW_1_trace.mat", "capture_78in_NW_2_trace.mat", "capture_78in_NW_4_trace.mat", "capture_78in_NW_8_trace.mat"],
    ["capture_82in_E_1_trace.mat", "capture_82in_E_2_trace.mat", "capture_82in_E_4_trace.mat", "capture_82in_E_8_trace.mat"]
]

css_ans = [4, 16, 1, 3, 19, 9, 11, 17, 13, 5]


home_batches = [
    ["1709_111in_NE_capture_1_trace.mat", "1709_111in_NE_capture_2_trace.mat", "1709_111in_NE_capture_4_trace.mat", "1709_111in_NE_capture_8_trace.mat"],
    ["1709_115in_S_capture_1_trace.mat", "1709_115in_S_capture_2_trace.mat", "1709_115in_S_capture_4_trace.mat", "1709_115in_S_capture_8_trace.mat"],
    ["1709_115in_SW_capture_1_trace.mat", "1709_115in_SW_capture_2_trace.mat", "1709_115in_SW_capture_4_trace.mat", "1709_115in_SW_capture_8_trace.mat"],
    ["1709_63in_E_capture_1_trace.mat", "1709_63in_E_capture_2_trace.mat", "1709_63in_E_capture_4_trace.mat", "1709_63in_E_capture_8_trace.mat"],
    ["capture_60in_W_1_trace.mat", "capture_60in_W_2_trace.mat", "capture_60in_W_4_trace.mat", "capture_60in_W_8_trace.mat"],
    ["capture_62in_S_1_trace.mat", "capture_62in_S_2_trace.mat", "capture_62in_S_4_trace.mat", "capture_62in_S_8_trace.mat"],
    ["capture_62in_SE_1_trace.mat", "capture_62in_SE_2_trace.mat", "capture_62in_SE_4_trace.mat", "capture_62in_SE_8_trace.mat"],
    ["capture_69in_N_1_trace.mat", "capture_69in_N_2_trace.mat", "capture_69in_N_4_trace.mat", "capture_69in_N_8_trace.mat"],
    ["capture_69in_SW_1_trace.mat", "capture_69in_SW_2_trace.mat", "capture_69in_SW_4_trace.mat", "capture_69in_SW_8_trace.mat"],
    ["capture_88in_NE_1_trace.mat", "capture_88in_NE_2_trace.mat", "capture_88in_NE_4_trace.mat", "capture_88in_NE_8_trace.mat"]

]

home_ans = [3, 16, 15, 4, 13, 8, 5, 6, 14, 3]

#css_ans -= 1
info_arr = [[10.6], [10.6], [3.3] ,[7.35]]
home_info_arr = [[6.9], [4.0], [2.1], [14.7]]
css_info_arr = [[8.6], [5.2], [3.6], [8.09]]
model = keras.models.load_model('cnn_model_equal_augmented_norm_fixed_98F1.h5', custom_objects={'leaky-relu': LeakyReLU()})
start = timeit.default_timer()
distance_wrong = 0
num_wrong = 0
for files, ans in zip(batches, batches_ans):
    result_arr = np.empty([1, 1024])
    for file in files:
      print(file)
      csi = preprocess_csi(extract_csi(test_dir + file))
      csi = np.array(np_csi_normalized(csi))
 #     csi = clean_image(csi)
      result_arr = np.vstack((result_arr, csi))
#    result_arr = clean_image(result_arr)
    result_arr = np.delete(result_arr, 0, axis=0)
    result_arr = np.hstack((info_arr, result_arr))
    result_arr = clean_image(result_arr)[:, :-1]
    result_arr = tf.expand_dims(result_arr, axis=0)
#    print(result_arr.shape)
#    print(result_arr)


#X, Y = load_unseen_data()
#Y -= 1
    prediction = model.predict(np.array(result_arr))
#    print(np.argmax(prediction, axis=1)[0])
    Y_true = ans
    if int(np.argmax(prediction, axis=1)[0]) != int(ans):
        print(str(np.argmax(prediction, axis=1)[0]) + "\t" + str(ans))
        num_wrong += 1
    else:
        print(str(np.argmax(prediction, axis=1)[0]) + "\t" + str(ans))
    distance_wrong += calculate_distance_error(np.argmax(prediction, axis=1)[0], Y_true)
print(distance_wrong/10)
stop = timeit.default_timer()
print("runtim: " + str(stop-start))
