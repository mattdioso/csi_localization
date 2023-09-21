#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

combined_dir = "/home/matt/hidden-device-research/datasets/combined_fixed"
file = combined_dir + "/bedroom.csv"
data_file = combined_dir + "/bedroom.npy"
data_mags_file = combined_dir + "/bedroom_mags.npy"
data_mags_norm_file = combined_dir + "/bedroom_mags_norm.npy"

df = pd.read_csv(file, header=None)
print(df.size)
#df.insert(8, column="8", value=0.1634)
#df.drop(df.columns[[0]], axis=1, inplace=True)
#df.to_csv(combined_dir + "/bedroom.csv", header=None, index=False)
print(df[5].unique())
a1 = df.loc[df[5] == 1].reset_index(drop=True).sort_values(4).reset_index(drop=True)
a2 = df.loc[df[5] == 2].reset_index(drop=True).sort_values(4).reset_index(drop=True)
a3 = df.loc[df[5] == 4].reset_index(drop=True).sort_values(4).reset_index(drop=True)
a4 = df.loc[df[5] == 8].reset_index(drop=True).sort_values(4).reset_index(drop=True)
print(a1.shape[0])
print(a2.shape[0])
print(a3.shape[0])
print(a4.shape[0])

length = a1.shape[0]
if a2.shape[0] < length:
  length = a2.shape[0]
if a3.shape[0] < length:
  length = a3.shape[0]
if a4.shape[0] < length:
  length = a4.shape[0]

print(length)

dataset = []
dataset_mags = []
dataset_mags_norm = []

env_arr = a1.loc[0, :8]
print(env_arr)
L = env_arr[1]
W = env_arr[2]
H = env_arr[3]
D = env_arr[8]
env_info = np.array([[L], [W], [H], [D]])
print(env_info)
