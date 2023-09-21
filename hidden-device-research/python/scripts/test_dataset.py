#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

combined_dir = "/Volumes/External Drive/research/mscse-capstone/hidden-device-research/datasets/combined"
file = combined_dir + "/321A.npy"
mags_file = combined_dir + "/321A_mags.npy"
mags_norm_file = combined_dir + "/321A_mags_norm.npy"

with open(file, 'rb') as f:
  np_arr = np.load(f, allow_pickle=True)

print(np_arr.shape)
print(np_arr[0])

with open(mags_file, 'rb') as f:
  np_arr = np.load(f, allow_pickle=True)

print(np_arr.shape)
print(np_arr[0])

with open(mags_norm_file, 'rb') as f:
  np_arr = np.load(f, allow_pickle=True)

print(np_arr.shape)
print(np_arr[10000])
