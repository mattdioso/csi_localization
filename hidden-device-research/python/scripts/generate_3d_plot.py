#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing as pre
np.set_printoptions(threshold=sys.maxsize)
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection="3d")

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

combined_dir = "/Volumes/External Drive/research/mscse-capstone/hidden-device-research/datasets/combined"
file = combined_dir + "/321A.csv"
data_file = combined_dir + "/321A.npy"
data_mags_file = combined_dir + "/321A_mags.npy"
data_mags_norm_file = combined_dir + "/321A_mags_norm.npy"

df = pd.read_csv(file, header=None)
a1 = df.loc[df[5] == 8].reset_index(drop=True).sort_values(4)


grid_1 = a1.loc[a1[4] == 1].reset_index(drop=True)
print(grid_1.shape)

grid_11 = a1[a1[4] == 11].reset_index(drop=True)
print(grid_11.shape)

grid_20 = a1[a1[4] == 20].reset_index(drop=True)
print(grid_20.shape)

grid_1_label_added = False
for i in range(200):
  csi = preprocess_csi(grid_1.loc[i, 9:].to_numpy())
  csi = np_csi_normalized(csi)
  #print(csi)
  x = np.array(range(0, len(csi)))
  if not grid_1_label_added:
    ax.plot3D(x, csi,i, color="red", label="Grid 1")
#    ax.scatter3D(x, csi, i, c=csi, cmap='cividis')
    grid_1_label_added = True
  else:
    ax.plot3D(x, csi, i, color="red")
#    ax.scatter3D(x, csi, i, c=csi, cmap='cividis')
"""
grid_11_label_added = False
for i in range(200):
  csi = preprocess_csi(grid_11.loc[i, 9:].to_numpy())
  csi = np_csi_normalized(csi)
  x = np.array(range(0, len(csi)))
  if not grid_11_label_added:
    ax.plot3D(x, csi, i, color="blue", label="Grid 11")
    grid_11_label_added = True
  else:
    ax.plot3D(x, csi, i, color="blue")

grid_20_label_added = False
for i in range(200):
  csi = preprocess_csi(grid_20.loc[i, 9:].to_numpy())
  csi = np_csi_normalized(csi)
  x = np.array(range(0, len(csi)))
  if not grid_20_label_added:
    ax.plot3D(x, csi, i, color="green", label="Grid 20")
    grid_20_label_added = True
  else:
    ax.plot3D(x, csi, i, color="green")
"""
plt.ylabel("Normalized Amplitude")
plt.xlabel("Subcarrier")
plt.legend(loc="upper left")
plt.show()
