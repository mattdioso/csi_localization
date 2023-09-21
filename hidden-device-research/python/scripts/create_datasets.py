#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import pickle

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

def np_csi_normalized(csi):
  csi_norm = (csi - np.min(csi))/(np.max(csi) - np.min(csi))
  return csi_norm

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

#label = np.array([[env_arr[4]], [env_arr[4]], [env_arr[4]], [env_arr[4]]])

for i in range(length):
  a1_r = a1.loc[i, 9:].to_numpy()
  a2_r = a2.loc[i, 9:].to_numpy()
  a3_r = a3.loc[i, 9:].to_numpy()
  a4_r = a4.loc[i, 9:].to_numpy()

  label = np.array([[a1.loc[i, 4]], [a2.loc[i, 4]], [a3.loc[i, 4]], [a4.loc[i, 4]]])

  csi_set = np.vstack((a1_r, a2_r, a3_r, a4_r))
  full_set = np.hstack((env_info, csi_set))
  full_set_label = np.hstack((full_set, label))
  #print(full_set_label.shape)
  #print(full_set_label)
  dataset.append(full_set_label)

  a1_mags_r = preprocess_csi(a1_r)
  a2_mags_r = preprocess_csi(a2_r)
  a3_mags_r = preprocess_csi(a3_r)
  a4_mags_r = preprocess_csi(a4_r)

  csi_mags_set = np.vstack((a1_mags_r, a2_mags_r, a3_mags_r, a4_mags_r))
  full_mags_set = np.hstack((env_info, csi_mags_set))
  full_mags_set_label = np.hstack((full_mags_set, label))
  dataset_mags.append(full_mags_set_label)

  a1_mags_norm_r = np_csi_normalized(a1_mags_r)
  a2_mags_norm_r = np_csi_normalized(a2_mags_r)
  a3_mags_norm_r = np_csi_normalized(a3_mags_r)
  a4_mags_norm_r = np_csi_normalized(a4_mags_r)

  csi_mags_norm_set = np.vstack((a1_mags_norm_r, a2_mags_norm_r, a3_mags_norm_r, a4_mags_norm_r))
  full_mags_norm_set = np.hstack((env_info, csi_mags_norm_set))
  full_mags_norm_set_label = np.hstack((full_mags_norm_set, label))
  dataset_mags_norm.append(full_mags_norm_set_label)

print(np.asarray(dataset).shape)
#np.asarray(dataset).tofile(data_file, sep=',')
with open(data_file, 'wb') as f:
  np.save(f, np.asarray(dataset), allow_pickle=True)

with open(data_mags_file, 'wb') as f:
  np.save(f, np.asarray(dataset_mags), allow_pickle=True)

with open(data_mags_norm_file, 'wb') as f:
  np.save(f, np.asarray(dataset_mags_norm), allow_pickle=True)
