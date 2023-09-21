#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
import os

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

combined_dir = "/home/matt/hidden-device-research/datasets/combined"
#file = combined_dir + "/320.csv"
data_file = combined_dir + "/320_augmented.npy"
data_mags_file = combined_dir + "/320_mags_augmented.npy"
data_mags_norm_file = combined_dir + "/320_mags_norm_augmented.npy"
file = data_mags_norm_file

with open(file, 'rb') as f:
  np_arr = np.load(f, allow_pickle=True)
for j in np_arr:
  cleaned_csi = clean_image(j)[:, 1:-1]
  print(cleaned_csi.max())

