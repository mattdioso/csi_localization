#!/usr/bin/env python3
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing as pre
from oct2py import octave
np.set_printoptions(threshold=sys.maxsize)
octave.addpath('../../matlab/')


def preprocess_csi(csi):
  csi = csi.astype(complex)
  csi_b = np.linalg.norm(csi)
  csi_norm = csi/csi_b

  mags =[]
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

def preprocess_csi_normalized(csi):
  csi = csi.astype(complex)
  csi_b = np.linalg.norm(csi)
  csi_norm = csi/csi_b

  mags = []
  phases = []
  for x in csi_norm:
    A = x.real
    B = x.imag

    square = np.square(A) + np.square(B)
    mag = np.square(square)
    phase = np.arctan(B/A)

    phases.append(phase)
    mags.append(mag)

  np_mags = np.array(mags)
  return np_mags

def np_csi_normalized(csi):
#  csi = csi.astype(complex)
  csi_norm = (csi - np.min(csi))/(np.max(csi) - np.min(csi))
  return csi_norm

def sklearn_csi_normalized(csi):
#  csi = csi.astype(complex)
  csi = csi.reshape(-1, 1)

  csi_norm = pre.MinMaxScaler().fit_transform(csi)
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

dense_mat = "/Volumes/External Drive/research/mscse-capstone/hidden-device-research/results/tests/bedroom/dense/mats/small_6_1_dense_trace.mat"
undense_mat = "/Volumes/External Drive/research/mscse-capstone/hidden-device-research/results/tests/bedroom/undense/mats/small_6_1_undense_trace.mat"

#dense_csi = np_csi_normalized(preprocess_csi(extract_csi(dense_mat)))
#undense_csi = np_csi_normalized(preprocess_csi(extract_csi(undense_mat)))

dense_csi = preprocess_csi(extract_csi(dense_mat))
undense_csi = preprocess_csi(extract_csi(undense_mat))

print(dense_csi.shape)
print(undense_csi.shape)

x = np.array(range(0, len(dense_csi)))
plt.plot(x, dense_csi, color='blue', label='Empty Room')
plt.plot(x, undense_csi, color='green', label='Dense Room')

plt.ylabel('Amplitude')
plt.xlabel('Subcarrier')
plt.legend(loc='upper left')
plt.show()
