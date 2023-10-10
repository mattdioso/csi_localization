#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing as pre

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

csv_1 = "./csv_files/test_trace_1.csv"
csv_2 = "./csv_files/test_trace_2.csv"

df = pd.read_csv(csv_1, header=None)
for i in range(50):
    csi = preprocess_csi(df.loc[i].to_numpy())
    csi = np_csi_normalized(csi)
    x = np.array(range(0, len(csi)))
    plt.plot(x, csi, color="red")

df = pd.read_csv(csv_2, header=None)
for i in range(50):
    csi = preprocess_csi(df.loc[i].to_numpy())
    csi = np_csi_normalized(csi)
    x = np.array(range(0, len(csi)))
    plt.plot(x, csi, color="blue")

plt.show()
